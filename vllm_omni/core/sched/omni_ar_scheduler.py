from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from time import time
from typing import Any

import numpy as np
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler as AsyncVLLMScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
    uses_full_payload_input_coordinator,
)
from vllm_omni.core.sched.utils import omni_routed_experts_for_request
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.outputs import OmniConnectorOutput

logger = init_logger(__name__)


class SampledLogprobContractError(RuntimeError):
    """The model runner returned unusable sampled-token logprobs."""


def _slice_sampled_logprobs(logprobs: Any, req_index: int, sampled_token_ids: list[int]) -> Any:
    """Slice and validate the sampled-token logprobs for one AR request."""
    if logprobs is None:
        raise SampledLogprobContractError("AR logprobs were requested, but the model runner returned none")

    sliced = logprobs.slice_request(req_index, len(sampled_token_ids))
    token_rows = np.asarray(sliced.logprob_token_ids)
    value_rows = np.asarray(sliced.logprobs)
    expected_rows = len(sampled_token_ids)

    if token_rows.ndim != 2 or value_rows.ndim != 2:
        raise SampledLogprobContractError(
            "AR sampled-token logprobs must be rank-2 arrays, "
            f"got token_ids={token_rows.shape} logprobs={value_rows.shape}"
        )
    if token_rows.shape[0] != expected_rows or value_rows.shape[0] != expected_rows:
        raise SampledLogprobContractError(
            "AR sampled-token logprob row count does not match generated tokens: "
            f"tokens={expected_rows} token_id_rows={token_rows.shape[0]} "
            f"logprob_rows={value_rows.shape[0]}"
        )
    if expected_rows == 0:
        return sliced
    if token_rows.shape[1] == 0 or value_rows.shape[1] == 0:
        raise SampledLogprobContractError("AR sampled-token logprob rows are empty")

    sampled = np.asarray(sampled_token_ids)
    if not np.array_equal(token_rows[:, 0], sampled):
        mismatch = np.flatnonzero(token_rows[:, 0] != sampled)
        first = int(mismatch[0])
        raise SampledLogprobContractError(
            "AR sampled-token logprobs are misaligned: "
            f"row={first} generated_token={int(sampled[first])} "
            f"logprob_token={int(token_rows[first, 0])}"
        )
    if not np.isfinite(value_rows[:, 0]).all():
        bad_rows = np.flatnonzero(~np.isfinite(value_rows[:, 0])).tolist()
        raise SampledLogprobContractError(f"AR sampled-token logprobs contain non-finite values at rows {bad_rows}")
    return sliced


class OmniARScheduler(OmniSchedulerMixin, VLLMScheduler):
    """Synchronous AutoRegressive scheduler for vLLM-Omni. This class is also
    used as a base class for the OmniARAsyncScheduler and holds most of the
    core scheduling logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track requests that need KV cache transfer when finished
        # Value is {"seq_len": int, "block_ids": list[int]}
        self.requests_needing_kv_transfer: dict[str, dict[str, Any]] = {}

        # Track requests waiting for KV transfer (blocks not freed yet)
        self.waiting_for_transfer_free: set[str] = set()

        # Track ACTIVE transfers (submitted to runner but not yet acked via kv_extracted_req_ids)
        self.active_kv_transfers: set[str] = set()

        # Requests marked for deferred stop: keep running until KV extraction
        # completes so that kv_ready can be emitted while the request is still
        # alive.  Stopped on the first scheduler step after extraction ack.
        self.pending_stop_after_extraction: set[str] = set()

        self.finished_req_ids_dict = defaultdict(set)

        # [Omni] Pre-parse KV transfer criteria
        self.kv_transfer_criteria = self._get_kv_transfer_criteria()

        # Track requests that have already triggered prefill transfer to avoid duplicates
        self.transfer_triggered_requests: set[str] = set()
        # Emit one producer-complete event so Orchestrator can submit the
        # decode worker before the consumer's KV extraction acknowledgement.
        self._pd_prefill_submit_ready_requests: set[str] = set()
        # The KV extraction acknowledgement can arrive in a later scheduler
        # step than the model output. Retain the Qwen3-TTS PD runtime payload
        # until it can be attached to that kv_ready event.
        self._kv_ready_multimodal_output_by_req: dict[str, dict[str, Any]] = {}

        # Cache per-request flag to avoid repeated deserialization of additional_information
        self._omits_kv_transfer_cache: dict[str, bool] = {}
        model_config = self.vllm_config.model_config
        self.chunk_transfer_adapter = None
        if getattr(model_config, "async_chunk", False):
            self.chunk_transfer_adapter = OmniChunkTransferAdapter(self.vllm_config)
        self.input_coordinator: OmniSchedulingCoordinator | None = None
        if uses_full_payload_input_coordinator(model_config):
            self.input_coordinator = OmniSchedulingCoordinator(
                stage_id=getattr(model_config, "stage_id", 0),
            )
        self._latest_omni_connector_output: OmniConnectorOutput | None = None
        # Snapshot prompt length for each streaming input update
        self._new_prompt_len_snapshot: dict[str, int] = {}

        # [Omni][PD] Never admit more decode work than the KV pool can hold.
        self._maybe_cap_running_reqs_for_pd_decode()

    def _get_confirmed_num_computed_tokens(self, request: Request) -> int:
        """num_computed_tokens minus async placeholders (KV actually on GPU)."""
        # Output placeholders are zero when async scheduling isn't used
        return request.num_computed_tokens - request.num_output_placeholders

    def _uses_native_pd_kv_transfer(self) -> bool:
        """Whether upstream vLLM's KV connector owns this P/D handoff."""
        kv_transfer_config = getattr(self.vllm_config, "kv_transfer_config", None)
        return getattr(kv_transfer_config, "kv_role", None) == "kv_producer"

    def _get_kv_transfer_criteria(self) -> dict | None:
        # Note: vllm_config is available in Scheduler after super().__init__
        if not hasattr(self, "vllm_config"):
            return None

        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            if isinstance(omni_kv_config, dict):
                criteria = omni_kv_config.get("kv_transfer_criteria", None)
            else:
                criteria = getattr(omni_kv_config, "kv_transfer_criteria", None)
            if criteria:
                return criteria
        # Mooncake P/D owns KV movement in its native connector, not in the
        # OmniKVTransferManager. It still needs a scheduler stop boundary.
        if self._uses_native_pd_kv_transfer():
            return {"type": "prefill_finished", "stop_after_transfer": True}
        return None

    def _request_omits_kv_transfer_to_next_stage(self, request: Request) -> bool:
        """True when this stage-zero-final request does not need downstream KV.

        The result is cached per request to avoid repeated deserialization of
        additional_information on every scheduler tick.
        """
        rid = request.request_id
        cached = self._omits_kv_transfer_cache.get(rid)
        if cached is not None:
            return cached

        payload = getattr(request, "additional_information", None)
        if payload is None:
            result = False
        else:
            info = deserialize_additional_information(payload)
            result = info.get("omni_final_stage_id") == 0 and not bool(info.get("omni_force_kv_transfer", False))

        self._omits_kv_transfer_cache[rid] = result
        return result

    def _should_defer_waiting_admission(self) -> bool:
        return False

    def _process_kv_transfer_trigger(self, request: Request, new_token_ids: list[int]) -> bool:
        """
        Check triggers and process side effects (marking transfer).
        Returns True if request should be STOPPED.
        Returns False if request should continue (even if transfer was triggered).
        """
        if not self.kv_transfer_criteria:
            return False

        # Text-only requests finalize at stage 0; do not prefill-stop for DiT KV.
        if self._request_omits_kv_transfer_to_next_stage(request):
            return False

        if request.request_id in self.waiting_for_transfer_free:
            return False

        criteria_type = self.kv_transfer_criteria.get("type")
        stop_decode_on_trigger = self.kv_transfer_criteria.get("stop_after_transfer", True)

        if request.request_id in self.transfer_triggered_requests:
            # Deferred stop: once KV extraction is complete (no longer in
            # active_kv_transfers), stop the request.  This guarantees the
            # kv_ready signal was emitted while the request was still alive.
            if (
                request.request_id in self.pending_stop_after_extraction
                and request.request_id not in self.active_kv_transfers
            ):
                self.pending_stop_after_extraction.discard(request.request_id)
                request.status = RequestStatus.FINISHED_STOPPED
                return True
            return False

        # seq_len for KV transfer must exclude async placeholders.
        confirmed_computed = self._get_confirmed_num_computed_tokens(request)

        if criteria_type == "prefill_finished":
            if confirmed_computed >= request.num_prompt_tokens:
                self.transfer_triggered_requests.add(request.request_id)
                if self._uses_native_pd_kv_transfer():
                    # Mooncake's vLLM connector exports and retains its own
                    # KV blocks in _connector_finished(). Do not also queue a
                    # CPU OmniKVTransferManager copy or treat its local return
                    # value as a consumer-pull acknowledgement.
                    self._pd_prefill_submit_ready_requests.add(request.request_id)
                    return bool(stop_decode_on_trigger)

                self._mark_request_for_kv_transfer(request.request_id, confirmed_computed)
                actually_queued = request.request_id in self.requests_needing_kv_transfer
                if actually_queued:
                    self._pd_prefill_submit_ready_requests.add(request.request_id)
                return bool(stop_decode_on_trigger and actually_queued)

        elif criteria_type == "special_token":
            target_token_id = self.kv_transfer_criteria.get("token_id")
            if target_token_id is not None and target_token_id in new_token_ids:
                self.transfer_triggered_requests.add(request.request_id)

                try:
                    idx = new_token_ids.index(target_token_id)
                    tokens_to_exclude = len(new_token_ids) - (idx + 1)
                    snapshot_len = confirmed_computed - tokens_to_exclude
                except ValueError:
                    snapshot_len = confirmed_computed

                if self._uses_native_pd_kv_transfer():
                    self._pd_prefill_submit_ready_requests.add(request.request_id)
                    return bool(stop_decode_on_trigger)

                self._mark_request_for_kv_transfer(request.request_id, snapshot_len)
                actually_queued = request.request_id in self.requests_needing_kv_transfer
                if actually_queued:
                    self._pd_prefill_submit_ready_requests.add(request.request_id)
                return bool(stop_decode_on_trigger and actually_queued)

        return False

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # Remove FINISHED_ABORTED requests before the upstream scheduler sees
        # them. Upstream vllm raises RuntimeError on this status; omni allows
        # async abort (e.g. client disconnect during TTS streaming) to leave
        # requests in the waiting/running queues temporarily.
        for queue in (self.waiting, self.running):
            for req in list(queue):
                if getattr(req, "status", None) == RequestStatus.FINISHED_ABORTED:
                    queue.remove(req)
        self._consume_pending_connector_output(model_mode="ar")
        self._process_pending_input_timeouts()
        # Recover a permanently wedged PD decode replica (running=0 with
        # un-schedulable preempted PD-consumer requests). No-op on all other
        # stages and whenever the replica is making progress.
        self._maybe_break_pd_decode_wedge()
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.process_pending_chunks(
                self.waiting, self.running, scheduler_requests=self.requests
            )

        original_waiting = None
        if self._should_defer_waiting_admission():
            original_waiting = self.waiting
            self.waiting = create_request_queue(self.policy)

        try:
            scheduler_output = super().schedule(throttle_prefills)
        finally:
            if original_waiting is not None:
                deferred_waiting = list(self.waiting)
                if deferred_waiting:
                    original_waiting.prepend_requests(deferred_waiting)
                self.waiting = original_waiting
            if self.chunk_transfer_adapter:
                # Add request waiting for chunk to the waiting and running queue
                self.chunk_transfer_adapter.restore_queues(
                    self.waiting,
                    self.running,
                    scheduler_requests=self.requests,
                )
            if self.input_coordinator:
                self.input_coordinator.restore_queues(self.waiting)
        try:
            # Late import to avoid circulars in some launch modes
            from .output import OmniNewRequestData

            # Rewrap base NewRequestData entries with OmniNewRequestData,
            # enriching with request-level payloads
            new_list = []
            for nr in scheduler_output.scheduled_new_reqs:
                req_id = getattr(nr, "req_id", None)
                request = self.requests.get(req_id) if req_id else None
                # Build omni entry preserving all base fields
                omni_nr = OmniNewRequestData(
                    req_id=nr.req_id,
                    external_req_id=(getattr(request, "external_req_id", None) if request else None),
                    prompt_token_ids=nr.prompt_token_ids,
                    mm_features=nr.mm_features,
                    sampling_params=nr.sampling_params,
                    pooling_params=nr.pooling_params,
                    block_ids=nr.block_ids,
                    num_computed_tokens=nr.num_computed_tokens,
                    lora_request=nr.lora_request,
                    # Enrich with omni payloads from the live request object
                    prompt_embeds=(getattr(request, "prompt_embeds", None) if request else None),
                    prompt_is_token_ids=nr.prompt_is_token_ids,
                    additional_information=(getattr(request, "additional_information", None) if request else None),
                    model_intermediate_buffer=(
                        getattr(request, "model_intermediate_buffer", None) if request else None
                    ),
                )
                new_list.append(omni_nr)

            scheduler_output.scheduled_new_reqs = new_list  # type: ignore[assignment]

            cached_reqs = scheduler_output.scheduled_cached_reqs
            cached_all_token_ids = dict(getattr(cached_reqs, "all_token_ids", {}) or {})
            for cached_req_id in getattr(cached_reqs, "req_ids", ()) or ():
                if cached_req_id in cached_all_token_ids:
                    continue
                cached_request = self.requests.get(cached_req_id)
                if cached_request is None:
                    continue
                live_all_token_ids = getattr(cached_request, "_all_token_ids", None)
                if live_all_token_ids is None:
                    continue
                cached_all_token_ids[cached_req_id] = list(live_all_token_ids)
            if cached_all_token_ids != getattr(cached_reqs, "all_token_ids", None):
                cached_reqs.all_token_ids = cached_all_token_ids

            if self.chunk_transfer_adapter:
                self.chunk_transfer_adapter.postprocess_scheduler_output(scheduler_output, self.requests)
            # Add information about requests needing KV cache transfer
            finished_reqs = self.get_finished_requests_needing_kv_transfer()
        except Exception:
            # If anything goes wrong, leave the original output unchanged
            init_logger(__name__).exception("Failed to wrap scheduled_new_reqs with OmniNewRequestData")
            finished_reqs = {}

        # Wrap in omni scheduler output to carry transfer metadata.
        return self._wrap_omni_scheduler_output(
            scheduler_output,
            finished_requests_needing_kv_transfer=finished_reqs,
        )

    def _is_pd_decode_consumer_stage(self) -> bool:
        """True on the decode (kv_consumer) side of a PD split."""
        cfg = getattr(self.vllm_config, "kv_transfer_config", None)
        return getattr(cfg, "kv_role", None) in ("kv_consumer", "kv_both")

    def _kv_safe_max_running_reqs(self) -> int | None:
        """Max concurrent max-length sequences this stage's KV pool can hold.

        Reuses upstream's group-aware concurrency helper -- the same one that
        logs "Maximum concurrency for N tokens per request" at startup -- so the
        derived cap matches the number already visible in the engine log and
        hybrid / multi-group / num_gpu_blocks_override layouts are handled for
        free. Falls back to flat blocks-per-request math (which is block-size
        invariant) when that helper or the KV cache config is unavailable.

        Returns None when capacity cannot be determined, meaning "do not cap".
        """
        kv_cache_config = getattr(self, "kv_cache_config", None)

        try:
            from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

            if kv_cache_config is not None and getattr(kv_cache_config, "kv_cache_groups", None):
                cap = int(get_max_concurrency_for_kv_cache_config(self.vllm_config, kv_cache_config))
                if cap >= 1:
                    return cap
        except Exception:
            logger.debug(
                "[Omni][PD] group-aware KV concurrency helper unavailable; falling back to block math.",
                exc_info=True,
            )

        max_model_len = int(getattr(self, "max_model_len", 0) or 0)
        block_size = int(getattr(self, "block_size", 0) or 0)
        num_blocks = int(getattr(kv_cache_config, "num_blocks", 0) or 0)
        if not num_blocks:
            num_blocks = int(getattr(self.cache_config, "num_gpu_blocks", 0) or 0)
        if max_model_len <= 0 or block_size <= 0 or num_blocks <= 0:
            return None
        blocks_per_req = -(-max_model_len // block_size)  # ceil
        return (num_blocks // blocks_per_req) or None

    def _maybe_cap_running_reqs_for_pd_decode(self) -> None:
        """Clamp the decode-stage admission gate to KV-safe concurrency.

        ``max_num_seqs`` is really a batch-*shape* knob: the worker uses it to
        size the persistent batch and pick CUDA-graph capture sizes. The
        scheduler separately uses ``max_num_running_reqs`` purely as an
        admission gate. When ``max_num_seqs`` exceeds what the KV pool can hold
        at ``max_model_len``, the scheduler over-admits and the RUNNING loop is
        forced to call ``_preempt_request`` to claw blocks back.

        On a PD *consumer* stage a preempted request is unrecoverable: its
        remote prefill KV was already released on the producer, and the
        Qwen3-TTS talker rebuilds each step's input embedding from all 16
        codebooks of the previous frame while ``Request`` only carries
        codebook 0 -- a resumed request re-prefills against a zeroed
        ``codes.audio`` placeholder and would emit garbage. Such requests pile
        up as PREEMPTED, upstream only admits WAITING work while
        ``not preempted_reqs``, and the replica wedges at ``running=0``.

        So cap admission at KV capacity here. Lowering only
        ``max_num_running_reqs`` is safe: upstream reads it just in the
        admission gate and in ``assert len(self.running) <=
        max_num_running_reqs``, which a lower value only makes easier to
        satisfy. Worker-side batch sizing reads ``max_num_seqs``, untouched.

        Inert at normal load -- it only binds under a pathological pileup.
        Override with ``VLLM_OMNI_PD_DECODE_MAX_RUNNING`` (<=0 disables).
        """
        if not self._is_pd_decode_consumer_stage():
            return

        configured = getattr(self, "max_num_running_reqs", 0)
        if not configured:
            return

        try:
            raw = (os.getenv("VLLM_OMNI_PD_DECODE_MAX_RUNNING", "") or "").strip()
            if raw:
                override = int(raw)
                if override <= 0:
                    logger.info(
                        "[Omni][PD] Decode admission cap disabled via "
                        "VLLM_OMNI_PD_DECODE_MAX_RUNNING=%s (max_num_running_reqs=%d).",
                        raw,
                        configured,
                    )
                    return
                cap = override
            else:
                kv_safe = self._kv_safe_max_running_reqs()
                if kv_safe is None:
                    return
                cap = kv_safe
            cap = max(1, min(cap, configured))
        except Exception:
            init_logger(__name__).exception(
                "[Omni][PD] Failed to compute decode admission cap; leaving max_num_running_reqs=%d unchanged.",
                configured,
            )
            return

        if cap >= configured:
            return

        self.max_num_running_reqs = cap
        logger.info(
            "[Omni][PD] Decode (kv_consumer) admission cap: max_num_running_reqs %d -> %d "
            "(max_model_len=%d; max_num_seqs stays %s for worker batch/CUDA-graph sizing). "
            "A preempted PD-consumer request can never be re-admitted (remote prefill KV "
            "released; multi-codebook talker state absent from Request token ids), so "
            "admitting beyond KV capacity risks wedging the replica at running=0. "
            "Override with VLLM_OMNI_PD_DECODE_MAX_RUNNING (<=0 disables).",
            configured,
            cap,
            getattr(self, "max_model_len", -1),
            getattr(self.scheduler_config, "max_num_seqs", "?"),
        )

    def _maybe_break_pd_decode_wedge(self) -> None:
        """Recover a permanently wedged PD decode replica.

        Root cause (observed under sustained load + over-generation runaways):
        the decode stage preempts running requests under transient KV pressure.
        A preempted PD-*consumer* request cannot be re-admitted -- its remote
        prefill KV was already released on the producer, and a request that
        generated near ``max_model_len`` tokens would need to recompute more
        than ``max_num_batched_tokens`` in one step (the talker prefill is not
        chunkable here), so ``schedule()`` can never re-admit it. Such requests
        pile up as PREEMPTED, ``running`` drains to 0, and the whole replica
        makes zero progress forever (GPU 0%), stalling every request behind it.
        Single-node has no remote-KV dependency, so its preempted requests
        recompute locally and this never happens.

        We detect the wedge (nothing running, but waiting-queue requests that
        cannot be scheduled) sustained across a grace window of scheduler
        steps, and abort the stuck requests so the client gets a terminal
        response and the replica resumes serving new requests. This converts a
        total replica hang into graceful per-request failure under extreme
        pressure.
        """
        if not self._is_pd_decode_consumer_stage():
            return
        try:
            running = len(self.running)
            stuck = [
                r
                for r in self.waiting
                if r.status in (RequestStatus.PREEMPTED, RequestStatus.WAITING)
            ]
            # Wedge signature: nothing running yet requests stranded in waiting.
            if running == 0 and stuck:
                self._pd_wedge_ticks = getattr(self, "_pd_wedge_ticks", 0) + 1
            else:
                self._pd_wedge_ticks = 0
                return
            # Grace window: the engine core sleeps ~1ms per idle step, so this
            # is a few seconds of *continuous* zero-progress-with-backlog, which
            # normal operation never sustains. Tunable via env.
            threshold = int(os.getenv("VLLM_OMNI_PD_DECODE_WEDGE_TICKS", "3000") or "3000")
            if self._pd_wedge_ticks < threshold:
                return
            stuck_ids = [r.request_id for r in stuck]
            logger.error(
                "[Omni][PD] Decode replica wedged: running=0 with %d stranded "
                "waiting requests for %d scheduler steps (preempted PD-consumer "
                "requests cannot recompute -- remote prefill KV released). "
                "Aborting %d stuck request(s) to recover the replica: %s",
                len(stuck_ids),
                self._pd_wedge_ticks,
                len(stuck_ids),
                stuck_ids[:8],
            )
            self.finish_requests(stuck_ids, RequestStatus.FINISHED_ABORTED)
            self._pd_wedge_ticks = 0
        except Exception:
            init_logger(__name__).exception("[Omni][PD] wedge-recovery check failed")


    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        mm_outputs = getattr(model_runner_output, "multimodal_outputs", None)
        inter_stage_outputs = getattr(model_runner_output, "inter_stage_outputs", None)
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats: CUDAGraphStat | None = model_runner_output.cudagraph_stats

        # [Omni] Mirror upstream Scheduler.update_from_output's deferred-free
        # drain. This method is a full reimplementation that never calls
        # super().update_from_output(), so the drain has to be repeated here.
        #
        # Upstream sets defer_block_free=True when max_concurrent_batches > 1
        # and the stage is a KV consumer -- exactly the Qwen3-TTS PD decode
        # stage (async_scheduling + kv_role: kv_consumer). In that mode
        # _free_request_blocks() does not return blocks to the pool; it pops
        # them onto self.deferred_frees behind a processed_step_seq fence, and
        # this is the ONLY site that advances the fence and drains the queue.
        # Omitting it leaks every KV block ever freed on the decode replica
        # (normal finishes, aborts and preemptions alike) for the lifetime of
        # the process -> monotonically rising KV pressure -> preemption -> a
        # permanently wedged replica, since a preempted PD-consumer request can
        # never be re-admitted.
        if getattr(self, "defer_block_free", False) and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        # Cache this step's per-request state before handling a possible
        # kv_extracted acknowledgement below. An extraction acknowledgement can
        # arrive on a no-forward step, so include its request IDs as well as
        # requests with scheduled tokens.
        kv_extracted_ids = list(getattr(model_runner_output, "kv_extracted_req_ids", None) or [])
        if mm_outputs is not None and self.kv_transfer_criteria:
            state_req_ids = set(num_scheduled_tokens) | set(kv_extracted_ids)
            for req_id in state_req_ids:
                req_index = model_runner_output.req_id_to_index.get(req_id)
                if req_index is None or req_index >= len(mm_outputs):
                    continue
                mm_output = mm_outputs[req_index]
                if isinstance(mm_output, dict) and mm_output:
                    self._kv_ready_multimodal_output_by_req[req_id] = mm_output
                    if req_id in kv_extracted_ids:
                        logger.info(
                            "[PD_TRACE] qwen3_tts_kv_ack_state_cached req=%s keys=%s",
                            req_id,
                            sorted(mm_output),
                        )

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Pre-process KV extraction acks so that the per-request loop below
        # can see up-to-date active_kv_transfers state and emit kv_ready
        # signals while requests are still alive (before any deferred stop).
        if kv_extracted_ids:
            for req_id in kv_extracted_ids:
                try:
                    self.active_kv_transfers.discard(req_id)
                    req = self.requests.get(req_id)
                    if req is not None and not req.is_finished():
                        pd_state = self._kv_ready_multimodal_output_by_req.pop(req_id, None)
                        logger.info(
                            "[PD_TRACE] qwen3_tts_kv_ready_emit req=%s has_state=%s",
                            req_id,
                            bool(pd_state),
                        )
                        outputs[req.client_index].append(
                            OmniEngineCoreOutput(
                                request_id=req_id,
                                new_token_ids=[],
                                multimodal_output=pd_state,
                                kv_transfer_params={"kv_ready": True},
                            )
                        )
                except Exception:
                    init_logger(__name__).exception("Failed to pre-process KV extraction for %s", req_id)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is not None:
                # vLLM 0.26: settle the in-flight tokens counted in schedule().
                # Must happen before the skips below — failed-KV-load and
                # already-finished requests were incremented too, and the two
                # readers (allocate_slots, _connector_finished) clamp with
                # max(0, computed - in_flight), so a leaked counter silently
                # freezes sliding-window block freeing.
                request.num_in_flight_tokens -= num_tokens_scheduled
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # Skip requests that were recovered from KV load failure
                continue
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or async scheduling).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []
            status_before_stop = request.status
            new_logprobs = None
            logprob_validation_failed = False

            # Validate before mutating request token state. A bad runner output
            # is request-local: terminate only this request and keep processing
            # the rest of the batch.
            if (
                generated_token_ids
                and request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
            ):
                try:
                    new_logprobs = _slice_sampled_logprobs(logprobs, req_index, generated_token_ids)
                except SampledLogprobContractError as exc:
                    logger.error("Invalid AR sampled-token logprobs for request %s: %s", req_id, exc)
                    request.status = RequestStatus.FINISHED_ERROR
                    request.stop_reason = str(exc)
                    request.resumable = False
                    generated_token_ids = []
                    logprob_validation_failed = True

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = logprob_validation_failed
            is_segment_finished = False
            finished = False
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            mm_output = mm_outputs[req_index] if mm_outputs else None
            inter_stage_output = inter_stage_outputs[req_index] if inter_stage_outputs else None
            kv_transfer_params = None
            finish_reason = None
            routed_experts = None

            # Check for stop and update request status.
            if new_token_ids:
                num_sampled_tokens = len(new_token_ids)
                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
                if new_logprobs is not None and len(new_token_ids) < num_sampled_tokens:
                    # A mid-step stop (e.g. spec-decode tokens sampled past
                    # EOS) trims new_token_ids after the validation slice
                    # above; re-slice so the emitted logprob rows stay 1:1
                    # with the emitted tokens, as upstream vLLM does by
                    # slicing after the trim.
                    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            # Evaluate transfer even if a model stop token/length limit ended
            # this step: prefill completion can coincide with that terminal
            # token, and its KV still must be handed to decode.
            if self._process_kv_transfer_trigger(request, new_token_ids):
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                if not struct_output_request.grammar.accept_tokens(req_id, new_token_ids):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. Terminating request.",
                        new_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            if stopped:
                if model_runner_output.routed_experts is not None:
                    routed_experts = omni_routed_experts_for_request(model_runner_output.routed_experts, request)

                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                is_segment_finished = not finished
                if finished:
                    request.resumable = False
                if not finished:
                    # for streaming input request only
                    if self.chunk_transfer_adapter:
                        if self.vllm_config.model_config.stage_id != 0:
                            # Downstream async-chunk stages receive real payloads from the
                            # connector. This update only resumes polling for the next segment.
                            self.chunk_transfer_adapter.segment_finished_requests.discard(request.request_id)
                    outstanding_async_tokens = request.num_output_placeholders
                    if outstanding_async_tokens > 0:
                        # Discard only outputs that are already in flight and
                        # roll back their optimistic computed-token accounting.
                        request.async_tokens_to_discard = outstanding_async_tokens
                        request.num_computed_tokens -= outstanding_async_tokens
                        request.num_output_placeholders = 0
                    request.spec_token_ids = []
                    request._output_token_ids.clear()
                if finished:
                    kv_transfer_params, _ = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                elif status_before_stop == RequestStatus.WAITING_FOR_CHUNK:
                    # In async chunk mode, request may be in either queue.
                    # Remove from both to avoid stale queue entries.
                    stopped_running_reqs.add(request)
                    stopped_preempted_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            # [PD] getattr-guarded: upstream unit tests drive update_from_output with
            # a SimpleNamespace scheduler double that never runs __init__, so this
            # PD-only state may be absent. Same defensive pattern as
            # getattr(self, "_inflight_prefills", set()) in _free_request().
            pd_submit_ready = getattr(self, "_pd_prefill_submit_ready_requests", None)
            if pd_submit_ready is not None and req_id in pd_submit_ready:
                submit_params = {
                    "pd_submit_ready": True,
                    "transfer_id": f"xfer-{req_id}",
                    "remote_request_id": req_id,
                }
                if kv_transfer_params is None:
                    kv_transfer_params = submit_params
                else:
                    kv_transfer_params = {**kv_transfer_params, **submit_params}
                pd_submit_ready.remove(req_id)
                logger.info(
                    "[PD_TRACE] qwen3_tts_prefill_submit_ready req=%s has_state=%s",
                    req_id,
                    bool(mm_output),
                )
            if new_token_ids or mm_output is not None or pooler_output is not None or kv_transfer_params or stopped:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    OmniEngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        multimodal_output=mm_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=request.take_prefill_stats(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                        is_segment_finished=is_segment_finished,
                        new_prompt_len_snapshot=self._new_prompt_len_snapshot.get(req_id, None),
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

            if self.chunk_transfer_adapter is not None and (
                inter_stage_output is not None or is_segment_finished or finished
            ):
                self.chunk_transfer_adapter.save_async(
                    inter_stage_output,
                    request,
                    is_segment_finished,
                )

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        # [Main] Handle failed KV load requests
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    OmniEngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )
                if self.chunk_transfer_adapter is not None:
                    self.chunk_transfer_adapter.cleanup_receiver(
                        request.request_id,
                    )

        # [Omni] Cleanup state for finished requests
        # getattr-guarded: see the note at the pd_submit_ready lookup above --
        # SimpleNamespace scheduler doubles in upstream tests skip __init__.
        pd_state_cache = getattr(self, "_kv_ready_multimodal_output_by_req", None)
        pd_ready_set = getattr(self, "_pd_prefill_submit_ready_requests", None)
        for req in stopped_running_reqs:
            if req.request_id not in self.waiting_for_transfer_free:
                if pd_state_cache is not None:
                    pd_state_cache.pop(req.request_id, None)
                if pd_ready_set is not None:
                    pd_ready_set.discard(req.request_id)
                if req.request_id in self.transfer_triggered_requests:
                    self.transfer_triggered_requests.remove(req.request_id)
                if req.request_id in self.active_kv_transfers:
                    self.active_kv_transfers.remove(req.request_id)
                self.pending_stop_after_extraction.discard(req.request_id)

        # Same for preempted
        for req in stopped_preempted_reqs:
            if req.request_id not in self.waiting_for_transfer_free:
                if pd_state_cache is not None:
                    pd_state_cache.pop(req.request_id, None)
                if pd_ready_set is not None:
                    pd_ready_set.discard(req.request_id)
                if req.request_id in self.transfer_triggered_requests:
                    self.transfer_triggered_requests.remove(req.request_id)
                if req.request_id in self.active_kv_transfers:
                    self.active_kv_transfers.remove(req.request_id)
                self.pending_stop_after_extraction.discard(req.request_id)

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if scheduler_kv_connector_stats is not None and not scheduler_kv_connector_stats.is_empty():
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        # FIXME: finished_req_ids_dict is unconditionally initialized as
        # defaultdict(set) in __init__ (not gated by include_finished_set).
        # This branch is therefore always eligible once any client_index is
        # populated; revisit when wiring streaming-only / upstream semantics.
        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                eco = engine_core_outputs.get(client_index)
                if eco is None:
                    eco = EngineCoreOutputs()
                    engine_core_outputs[client_index] = eco
                emitted = {o.request_id for o in eco.outputs}
                for req_id in finished_set:
                    if req_id not in emitted:
                        eco.outputs.append(EngineCoreOutput(req_id, [], finish_reason=FinishReason.ABORT))
                eco.finished_requests = finished_set
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        self._capture_omni_connector_output(model_runner_output)

        # Free blocks that were held for transfer (kv_ready and
        # active_kv_transfers updates already done before the per-request loop).
        if kv_extracted_ids:
            for req_id in kv_extracted_ids:
                try:
                    if req_id in self.waiting_for_transfer_free:
                        req = self.requests.get(req_id)
                        if req:
                            self.kv_cache_manager.free(req)
                            if req_id in self.requests:
                                del self.requests[req_id]
                            if req_id in self.transfer_triggered_requests:
                                self.transfer_triggered_requests.remove(req_id)
                            self.active_kv_transfers.discard(req_id)
                            self.pending_stop_after_extraction.discard(req_id)
                            logger.debug(f"Freed blocks for {req_id} after transfer extraction")
                        self.waiting_for_transfer_free.remove(req_id)
                except Exception:
                    init_logger(__name__).exception("Failed to free blocks for %s after transfer", req_id)

        return engine_core_outputs

    def finish_requests(self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            The Request objects that were aborted. Will not include any that
            were already finished.
        """
        # TODO(yrr): chunk transfer adapter & input_coordinator unified to one
        if self.chunk_transfer_adapter:
            self.chunk_transfer_adapter.finish_requests(request_ids, finished_status, self.requests)

        # Realign stale ``request.status`` (chunk-transfer-adapter's
        # ``requests_origin_status`` table doesn't follow the
        # ``waiting → running`` admit transition; without this, an abort
        # arriving between admit and the next deque round-trip leaves
        # the request in ``self.running`` with ``status=WAITING`` and
        # upstream ``Scheduler.finish_requests`` silently fails to
        # release the worker's ``input_batch`` slot -- after
        # ``max_num_seqs`` such aborts new requests hang at
        # ``chunks=0``). Only the ``async_chunk`` path triggers the
        # staleness; with ``async_chunk`` disabled this is a cheap O(n)
        # no-op over an already-aligned set, kept unconditional so the
        # abort path stays uniform across configurations. See
        # ``OmniSchedulerMixin._realign_request_status_to_queues`` and
        # #3774 discussion.
        self._realign_request_status_to_queues(request_ids)

        finished = super().finish_requests(request_ids, finished_status)

        # Defensive post-finish purge: belt-and-suspenders to the
        # realignment above. Even after realign + ``super()``, corner
        # cases (mid-transition status, connector cleanups that pop
        # from ``self.requests`` without unwinding ``self.running``)
        # can leave already-finished or untracked entries in
        # ``self.running``. Sweep them now so the worker's
        # ``input_batch`` slot never pins a freed request and starves
        # new admissions. See ``OmniSchedulerMixin._purge_finished_from_running``.
        self._purge_finished_from_running()

        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is not None:
            for request in finished:
                self._free_input_coordinator_request(request.request_id)
        return finished

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """
        Override: Only extend prompt at stage 0, and replace
        the existing session with the next streaming update at other stages.

        Discards the last sampled output token from the prior input chunk at stage 0.
        """
        req_id = session.request_id
        self._new_prompt_len_snapshot[req_id] = len(update.prompt_token_ids)
        outstanding_async_tokens = getattr(session, "num_output_placeholders", 0)
        if outstanding_async_tokens > 0:
            # Async scheduling may already have sampled the previous
            # segment's next token. Drop that late token instead of
            # appending it to the new streaming segment.
            session.async_tokens_to_discard = 1
            session.num_computed_tokens -= session.num_output_placeholders
            session.num_output_placeholders = 0
            session.spec_token_ids = []
        stage_id = self.vllm_config.model_config.stage_id
        if self.chunk_transfer_adapter and self.chunk_transfer_adapter.receives_chunks:
            self.chunk_transfer_adapter.requests_num_chunks_sent.pop(session.external_req_id, None)
            if stage_id != 0:
                # Downstream async-chunk stages receive real payloads from the
                # connector. This update only resumes polling for the next segment.
                self.chunk_transfer_adapter.segment_finished_requests.discard(session.request_id)
                # Do not replace prompt/additional_information here; the next
                # upstream chunk will populate them in chunk transfer adapter.
                session.arrival_time = update.arrival_time
                session.sampling_params = update.sampling_params
                if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                session.status = RequestStatus.WAITING
                if session in self.skipped_waiting:
                    self.skipped_waiting.remove_requests((session,))
                    self._enqueue_waiting_request(session)

                if self.log_stats:
                    session.record_event(EngineCoreEventType.QUEUED)
                return
        update_infos = (
            getattr(update, "model_intermediate_buffer", None),
            getattr(update, "additional_information", None),
        )
        replace_streaming_prompt = any(
            isinstance(info, dict)
            and isinstance(info.get("meta"), dict)
            and info["meta"].get("replace_streaming_prompt") is True
            for info in update_infos
        )
        if replace_streaming_prompt:
            self._replace_streaming_session(session, update)
            return
        super()._update_request_as_session(session, update)
        if hasattr(update, "model_intermediate_buffer"):
            session.model_intermediate_buffer = update.model_intermediate_buffer

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        # TODO(wzliu)! for offline mode, we should not end process until all data is transferred
        """Mark a request as finished and free its resources."""
        assert request.is_finished()

        self._omits_kv_transfer_cache.pop(request.request_id, None)

        # [Upstream compat] Discard request from in-flight prefills set added
        # upstream for routed-experts in-flight reservation tracking.
        # Use getattr for safety with test __new__ code paths.
        getattr(self, "_inflight_prefills", set()).discard(request)

        # 1. Standard cleanup parts from base _free_request
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        self._new_prompt_len_snapshot.pop(request_id, None)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # Mirror the generation scheduler's try/finally pattern so the
        # input_coordinator entry is always pruned along every return path,
        # including the early returns for in-flight / waiting KV transfers
        # below. _free_input_coordinator_request is a no-op when the
        # coordinator is None, so the unconditional finally is safe.
        try:
            # 2. Omni Specific: Check if we need to transfer KV
            if self._should_transfer_kv_for_request(request_id):
                already_triggered = request_id in self.transfer_triggered_requests
                is_active = request_id in self.active_kv_transfers

                if already_triggered:
                    if is_active or request_id in self.requests_needing_kv_transfer:
                        # The snapshot has either been sent to the runner or is
                        # queued for its next no-forward transfer step. Retain
                        # blocks and request metadata until the extraction ACK.
                        logger.debug(f"[Omni] Request {request_id} finished with KV transfer pending. Waiting.")
                        self.waiting_for_transfer_free.add(request_id)
                        # [PD] Do NOT clear kv_xfer_params here: the native
                        # Mooncake connector's params (bootstrap addr/port)
                        # must reach the decode replica. Upstream 0.26 widened
                        # the return to (kv_xfer_params, ec_xfer_params); there
                        # are no encoder-cache params on this path.
                        return kv_xfer_params, None
                    elif request_id in self.waiting_for_transfer_free:
                        # Blocks held until KV extraction completes in a future step.
                        return None, None
                    else:
                        logger.debug(
                            f"[Omni] Request {request_id} finished and transfer no longer ACTIVE (extracted/acked). "
                            "Freeing immediately."
                        )
                else:
                    self.waiting_for_transfer_free.add(request_id)
                    confirmed_computed = self._get_confirmed_num_computed_tokens(request)
                    self._mark_request_for_kv_transfer(request_id, confirmed_computed)
                    # Return KV transfer metadata so it propagates to RequestOutput
                    if request_id in self.requests_needing_kv_transfer:
                        transfer_data = self.requests_needing_kv_transfer[request_id]
                        kv_xfer_params = {
                            "past_key_values": transfer_data["block_ids"],
                            "kv_metadata": {
                                "seq_len": transfer_data["seq_len"],
                                "block_ids": transfer_data["block_ids"],
                            },
                        }
                        # Also update request.additional_information for good measure
                        add_info = getattr(request, "additional_information", None)
                        # If additional_information is an AdditionalInformationPayload-like object,
                        # unpack it into a plain dict.
                        if (
                            add_info is not None
                            and hasattr(add_info, "entries")
                            and isinstance(getattr(add_info, "entries"), dict)
                        ):
                            request.additional_information = deserialize_additional_information(add_info)
                            add_info = request.additional_information
                        if add_info is None:
                            request.additional_information = {}
                            add_info = request.additional_information
                        if isinstance(add_info, dict):
                            add_info.update(kv_xfer_params)

                    return kv_xfer_params, None

            # 3. Standard Freeing
            delay_free_blocks |= connector_delay_free_blocks
            if not delay_free_blocks:
                self._free_blocks(request)

            return kv_xfer_params, None
        finally:
            self._free_input_coordinator_request(request_id)
            # Normal completion runs through here, not finish_requests()
            # (the abort path) -- see vllm-project/vllm-omni#5349.
            if self.chunk_transfer_adapter is not None:
                self.chunk_transfer_adapter.cleanup_receiver(request_id)

    def _update_from_kv_xfer_finished(self, kv_connector_output) -> None:
        """Guarded variant of the upstream KV-transfer finish handler.

        Upstream ``Scheduler._update_from_kv_xfer_finished`` asserts that every
        ``finished_recving`` / ``finished_sending`` request id is still present
        in ``self.requests``. That invariant does not hold in the Omni PD flow:
        a decode (consumer) request can be aborted (client disconnect) or a
        producer request can be reaped by Mooncake's own
        ``VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`` ("timed out ... without being
        sent") *before* the connector delivers the matching finished
        notification. When that late notification arrives the base assert fires
        inside ``update_from_output`` and kills the whole StageEngineCoreProc
        (EngineDeadError -> orchestrator dies -> server exits) — i.e. one
        stranded transfer escalates into a full-server outage.

        Here we skip ids that are no longer tracked (their blocks were already
        freed on the abort/cleanup path) and only free blocks for requests we
        still own, so a stranded transfer stays a per-request no-op instead of a
        fatal error. Behaviour for live requests is identical to upstream.
        """
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        for req_id in kv_connector_output.finished_recving or ():
            req = self.requests.get(req_id)
            if req is None:
                logger.warning(
                    "[Omni][PD] finished_recving for untracked req %s "
                    "(aborted/reaped before KV recv completed); skipping.",
                    req_id,
                )
                continue
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            elif RequestStatus.is_finished(req.status):
                self._free_blocks(req)
            else:
                logger.warning(
                    "[Omni][PD] finished_recving for req %s in unexpected "
                    "status %s; skipping free.",
                    req_id,
                    req.status,
                )

        for req_id in kv_connector_output.finished_sending or ():
            req = self.requests.get(req_id)
            if req is None:
                logger.warning(
                    "[Omni][PD] finished_sending for untracked req %s "
                    "(aborted/reaped before KV send completed); skipping.",
                    req_id,
                )
                continue
            self._free_blocks(req)

    def _mark_request_for_kv_transfer(self, req_id: str, seq_len: int) -> None:
        """Mark a request as needing KV cache transfer when it finishes."""
        # Avoid duplicate marking (if already pending in queue)
        if req_id in self.requests_needing_kv_transfer:
            return

        if self._should_transfer_kv_for_request(req_id):
            # [Omni] Get block IDs from KVCacheManager
            try:
                block_ids_tuple = self.kv_cache_manager.get_block_ids(req_id)
                if block_ids_tuple and len(block_ids_tuple) > 0:
                    block_ids = block_ids_tuple[0]

                    # [Omni] Fix: Truncate blocks to match seq_len snapshot
                    # We need to know block_size. Usually in self.cache_config.block_size
                    # Note: vllm_config might not be directly available, check scheduler_config or cache_config
                    if hasattr(self, "cache_config") and hasattr(self.cache_config, "block_size"):
                        block_size = self.cache_config.block_size
                    elif hasattr(self, "scheduler_config") and hasattr(
                        self.scheduler_config, "block_size"
                    ):  # Some versions
                        block_size = self.scheduler_config.block_size
                    else:
                        raise ValueError("Block size not found in cache_config or scheduler_config")

                    # ceil(seq_len / block_size)
                    num_blocks = (seq_len + block_size - 1) // block_size
                    if len(block_ids) > num_blocks:
                        logger.debug(
                            f"[Omni] Truncating blocks for {req_id} from {len(block_ids)} "
                            f"to {num_blocks} (seq_len={seq_len})"
                        )
                        block_ids = block_ids[:num_blocks]

                else:
                    block_ids = []
            except Exception as e:
                init_logger(__name__).warning(f"Failed to get block IDs for {req_id}: {e}")
                block_ids = []

            self.requests_needing_kv_transfer[req_id] = {"seq_len": seq_len, "block_ids": block_ids}
            logger.debug(f"Marked request {req_id} for KV cache transfer (len={seq_len}, blocks={len(block_ids)})")

    def _should_transfer_kv_for_request(self, req_id: str) -> bool:
        """Determine if a request should trigger KV cache transfer."""
        if self._uses_native_pd_kv_transfer():
            request = self.requests.get(req_id)
            return request is not None and not self._request_omits_kv_transfer_to_next_stage(request)

        need_send = False
        # Try to read from vLLM Config (where YAML config is typically loaded)
        # Check for omni_kv_config attribute
        omni_kv_config = getattr(self.vllm_config.model_config, "omni_kv_config", None)
        if omni_kv_config:
            # omni_kv_config could be an object or a dict
            if isinstance(omni_kv_config, dict):
                need_send = omni_kv_config.get("need_send_cache", False)
            else:
                need_send = getattr(omni_kv_config, "need_send_cache", False)
        if not need_send:
            return False
        request = self.requests.get(req_id)
        if request is not None and self._request_omits_kv_transfer_to_next_stage(request):
            return False
        return True

    def has_requests(self) -> bool:
        """Check if there are any requests to process, including KV transfers."""
        # [Omni] Also check for pending KV transfers
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_requests()

    def has_finished_requests(self) -> bool:
        """Check if there are any finished requests (including those needing KV transfer)."""
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_finished_requests()

    def has_unfinished_requests(self) -> bool:
        """Check if there are any unfinished requests (including those needing KV transfer)."""
        # [Omni] Also check for pending KV transfers to ensure the engine loop continues
        # MUST verify waiting_for_transfer_free and active_kv_transfers
        # Otherwise engine loop might exit before transfer Ack is received.
        if self.requests_needing_kv_transfer or self.active_kv_transfers or self.waiting_for_transfer_free:
            return True
        return super().has_unfinished_requests()

    def get_finished_requests_needing_kv_transfer(self) -> dict[str, dict]:
        """Get and clear the list of requests needing KV cache transfer.
        Returns dict: {req_id: {"seq_len": int, "block_ids": list[int]}}
        """
        requests = self.requests_needing_kv_transfer.copy()

        # Mark these requests as ACTIVE (sent to runner)
        self.active_kv_transfers.update(requests.keys())

        self.requests_needing_kv_transfer.clear()
        return requests


class OmniARAsyncScheduler(OmniARScheduler, AsyncVLLMScheduler):
    """Asynchronous AutoRegressive scheduler."""

