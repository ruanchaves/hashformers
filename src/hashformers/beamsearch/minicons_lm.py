import gc
import logging
import math
import warnings

import torch
from minicons import scorer

from hashformers.batching import (
    AUTO_BATCH_SIZE,
    DEFAULT_AUTO_BATCH_SIZE,
    DEFAULT_MAX_BATCH_SIZE,
    validate_batch_size,
    validate_max_batch_size,
)


LOGGER = logging.getLogger(__name__)

MIN_THROUGHPUT_IMPROVEMENT = 0.05
MIN_MEMORY_HEADROOM = 0.20


def ensure_tokenizer_compatibility(tokenizer):
    """Provide the tokenizer API expected by the minicons 0.3.39 wheel.

    Args:
        tokenizer: Hugging Face tokenizer used by the minicons scorer.

    Returns:
        The tokenizer with a compatible ``batch_encode_plus`` method.
    """
    if not callable(getattr(tokenizer, "batch_encode_plus", None)):
        tokenizer.batch_encode_plus = tokenizer.__call__
    return tokenizer


class MiniconsLM(object):
    """Score candidate sequences using lower-is-better costs.

    Integer ``gpu_batch_size`` values retain fixed microbatching. Passing
    ``"auto"`` opts a CUDA scorer into throughput-based growth from 64 up to
    ``max_gpu_batch_size``. Adaptive state belongs to this scorer instance, so
    a segmenter and reranker tune independently.

    Attributes:
        scorer: The minicons scorer used for the selected model type.
        gpu_batch_size: Configured fixed size or ``"auto"``.
        max_gpu_batch_size: Upper bound used only by automatic mode.
        effective_gpu_batch_size: Microbatch size currently selected.
        model_type: Name of the configured minicons scorer type.
    """

    def __init__(
        self,
        model_name_or_path,
        device="cuda",
        gpu_batch_size=20,
        model_type="IncrementalLMScorer",
        max_gpu_batch_size=DEFAULT_MAX_BATCH_SIZE,
    ):
        validate_batch_size(gpu_batch_size)
        validate_max_batch_size(max_gpu_batch_size)
        self.scorer = getattr(scorer, model_type)(model_name_or_path, device)
        ensure_tokenizer_compatibility(self.scorer.tokenizer)
        self.model_type = model_type
        self._configure_batching(device, gpu_batch_size, max_gpu_batch_size)

    def _configure_batching(self, device, gpu_batch_size, max_gpu_batch_size):
        """Initialize process-local batching state for this scorer."""
        validate_batch_size(gpu_batch_size)
        validate_max_batch_size(max_gpu_batch_size)
        self.device = str(device)
        self.gpu_batch_size = gpu_batch_size
        self.max_gpu_batch_size = max_gpu_batch_size
        self.effective_gpu_batch_size = (
            min(DEFAULT_AUTO_BATCH_SIZE, max_gpu_batch_size)
            if gpu_batch_size == AUTO_BATCH_SIZE
            else gpu_batch_size
        )
        self._auto_cuda = (
            gpu_batch_size == AUTO_BATCH_SIZE
            and self.device.startswith("cuda")
            and torch.cuda.is_available()
        )
        self._tuning_state = (
            "warming"
            if self._auto_cuda
            else "cpu"
            if gpu_batch_size == AUTO_BATCH_SIZE
            else "fixed"
        )
        self._baseline_throughput = None
        self._failed_upper_bound = None
        self._last_known_safe_batch_size = None
        self._observed_throughput = None
        self._peak_memory_bytes = None
        self._free_memory_bytes = None
        self._total_memory_bytes = None
        self._memory_headroom = None
        self._oom_backoff_events = 0

    @property
    def batch_telemetry(self):
        """Return a snapshot of this scorer's microbatch controller state."""
        return {
            "configured_batch_size": self.gpu_batch_size,
            "effective_batch_size": self.effective_gpu_batch_size,
            "max_batch_size": self.max_gpu_batch_size,
            "tuning_state": self._tuning_state,
            "candidates_per_second": self._observed_throughput,
            "peak_memory_bytes": self._peak_memory_bytes,
            "free_memory_bytes": self._free_memory_bytes,
            "total_memory_bytes": self._total_memory_bytes,
            "memory_headroom": self._memory_headroom,
            "oom_backoff_events": self._oom_backoff_events,
            "failed_upper_bound": self._failed_upper_bound,
        }

    def get_probs(self, list_of_candidates):
        """Score candidates in fixed or adaptively selected microbatches.

        Adaptive OOM recovery advances the cursor only after a whole slice has
        succeeded, which preserves candidate order without gaps or duplicates.
        """
        candidates = list(list_of_candidates)
        if not candidates:
            return []
        if not self._auto_cuda:
            return self._score_fixed(candidates)

        probabilities = []
        cursor = 0
        while cursor < len(candidates):
            remaining = len(candidates) - cursor
            batch_size, measurement = self._next_adaptive_batch(remaining)
            batch = candidates[cursor:cursor + batch_size]
            retry_after_oom = False
            try:
                if measurement is None:
                    scores = self.get_batch_scores(batch)
                    metrics = None
                else:
                    scores, metrics = self._score_timed_batch(batch)
            except RuntimeError as error:
                if not self._is_cuda_oom(error):
                    raise
                if batch_size <= 1:
                    raise
                retry_after_oom = True
            if retry_after_oom:
                # Run cache cleanup after leaving the exception handler so its
                # traceback no longer retains failed forward-pass tensors.
                self._recover_from_oom(batch_size)
                continue

            probabilities.extend(scores)
            cursor += batch_size
            self._last_known_safe_batch_size = max(
                self._last_known_safe_batch_size or 0, batch_size
            )
            if metrics is not None:
                self._record_measurement(batch_size, measurement, metrics)
        return probabilities

    def _score_fixed(self, candidates):
        """Score with the historical explicit-size behavior and no tuning."""
        probabilities = []
        for cursor in range(0, len(candidates), self.effective_gpu_batch_size):
            batch = candidates[cursor:cursor + self.effective_gpu_batch_size]
            probabilities.extend(self.get_batch_scores(batch))
        return probabilities

    def _next_adaptive_batch(self, remaining):
        """Choose a complete current batch, a growth trial, or an untimed tail."""
        effective = self.effective_gpu_batch_size
        if self._tuning_state == "converged":
            return min(effective, remaining), None

        if self._baseline_throughput is None:
            if remaining >= effective:
                self._tuning_state = "tuning"
                return effective, "baseline"
            return remaining, None

        trial_size = min(effective * 2, self.max_gpu_batch_size)
        if self._failed_upper_bound is not None:
            trial_size = min(trial_size, self._failed_upper_bound - 1)
        if trial_size <= effective:
            self._converge("configured or safe upper bound reached")
            return min(effective, remaining), None
        if remaining >= trial_size:
            return trial_size, "growth"
        return min(effective, remaining), None

    def _score_timed_batch(self, batch):
        """Synchronously time one CUDA microbatch and collect memory metrics."""
        with torch.cuda.device(self.device):
            torch.cuda.reset_peak_memory_stats(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            scores = self.get_batch_scores(batch)
            end.record()
        torch.cuda.synchronize(self.device)
        elapsed_seconds = max(start.elapsed_time(end) / 1000.0, 1e-12)
        free_memory, total_memory = torch.cuda.mem_get_info(self.device)
        peak_memory = torch.cuda.max_memory_allocated(self.device)
        return scores, {
            "throughput": len(batch) / elapsed_seconds,
            "peak_memory": peak_memory,
            "free_memory": free_memory,
            "total_memory": total_memory,
            "memory_headroom": free_memory / total_memory if total_memory else 0.0,
        }

    def _record_measurement(self, batch_size, measurement, metrics):
        """Update telemetry and accept or reject a geometric growth trial."""
        throughput = metrics["throughput"]
        headroom = metrics["memory_headroom"]
        self._observed_throughput = throughput
        self._peak_memory_bytes = metrics["peak_memory"]
        self._free_memory_bytes = metrics.get("free_memory")
        self._total_memory_bytes = metrics.get("total_memory")
        self._memory_headroom = headroom

        if measurement == "baseline":
            self._baseline_throughput = throughput
            if headroom < MIN_MEMORY_HEADROOM:
                self._converge("memory headroom is below 20%")
            elif self.effective_gpu_batch_size >= self.max_gpu_batch_size:
                self._converge("configured maximum reached")
            return

        improvement = throughput / self._baseline_throughput - 1.0
        if improvement < MIN_THROUGHPUT_IMPROVEMENT:
            self._converge("throughput improvement is below 5%")
            return
        if headroom < MIN_MEMORY_HEADROOM:
            self._converge("memory headroom is below 20%")
            return

        self.effective_gpu_batch_size = batch_size
        self._baseline_throughput = throughput
        if batch_size >= self.max_gpu_batch_size:
            self._converge("configured maximum reached")

    def _converge(self, reason):
        """Stop tuning and avoid further explicit CUDA synchronization."""
        if self._tuning_state != "converged":
            LOGGER.info(
                "Adaptive microbatching converged at %d for %s (%s)",
                self.effective_gpu_batch_size,
                self.model_type,
                reason,
            )
        self._tuning_state = "converged"

    def _recover_from_oom(self, failed_batch_size):
        """Back off after a CUDA OOM without consuming the failed slice."""
        self._failed_upper_bound = (
            failed_batch_size
            if self._failed_upper_bound is None
            else min(self._failed_upper_bound, failed_batch_size)
        )
        known_safe = self._last_known_safe_batch_size
        if known_safe is not None and known_safe < failed_batch_size:
            retry_size = known_safe
        else:
            retry_size = max(1, failed_batch_size // 2)
        self.effective_gpu_batch_size = min(
            self.effective_gpu_batch_size, retry_size
        )
        self._baseline_throughput = None
        self._oom_backoff_events += 1
        self._tuning_state = "converged"
        gc.collect()
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        LOGGER.warning(
            "CUDA OOM scoring %d candidates with %s; retrying the same slice "
            "with microbatches of %d",
            failed_batch_size,
            self.model_type,
            self.effective_gpu_batch_size,
        )

    @staticmethod
    def _is_cuda_oom(error):
        """Return whether ``error`` is specifically PyTorch's CUDA OOM type."""
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        return oom_type is not None and isinstance(error, oom_type)

    def incremental_sequence_score(self, batch):
        tokens = self.scorer.prepare_text(batch, bos_token=True, eos_token=True)
        stats = self.scorer.compute_stats(tokens, prob=True)
        log_stats = [[math.log(x) for x in sequence] for sequence in stats]
        sum_log_stats = [sum(x) for x in log_stats]
        return [1 - x for x in sum_log_stats]

    def get_batch_scores(self, batch):
        if self.model_type == "IncrementalLMScorer":
            return self.incremental_sequence_score(batch)
        if self.model_type == "MaskedLMScorer":
            return self.scorer.sequence_score(
                batch, reduction=lambda x: -x.sum(0).item()
            )
        if self.model_type == "Seq2SeqScorer":
            scores = self.scorer.sequence_score(batch, source_format="blank")
            return [-score for score in scores]
        warnings.warn(
            f"Model type {self.model_type} not implemented. Assuming negative "
            "summed log probability."
        )
        return self.scorer.sequence_score(
            batch, reduction=lambda x: -x.sum(0).item()
        )
