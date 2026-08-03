from contextlib import nullcontext

import sys

import pytest

from hashformers.beamsearch import minicons_lm
from hashformers.beamsearch.minicons_lm import MiniconsLM


def build_lm(monkeypatch, batch_size="auto", max_batch_size=512, device="cuda"):
    """Build a model-free scorer with initialized microbatch state."""
    monkeypatch.setattr(minicons_lm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        minicons_lm.torch.cuda, "device", lambda _device: nullcontext()
    )
    lm = object.__new__(MiniconsLM)
    lm.model_type = "TestScorer"
    lm._configure_batching(device, batch_size, max_batch_size)
    return lm


def install_recording_scores(lm, measurements=None):
    """Install deterministic scoring and optional synthetic CUDA metrics."""
    calls = []
    measurements = measurements or {}

    def score(batch):
        calls.append(("ordinary", list(batch)))
        return list(batch)

    def score_timed(batch):
        calls.append(("timed", list(batch)))
        throughput, headroom = measurements[len(batch)]
        return list(batch), {
            "throughput": throughput,
            "peak_memory": len(batch) * 1000,
            "free_memory": int(headroom * 1000),
            "total_memory": 1000,
            "memory_headroom": headroom,
        }

    lm.get_batch_scores = score
    lm._score_timed_batch = score_timed
    return calls


def test_auto_grows_geometrically_to_configured_maximum(monkeypatch):
    lm = build_lm(monkeypatch, max_batch_size=256)
    calls = install_recording_scores(
        lm,
        {64: (100.0, 0.80), 128: (110.0, 0.70), 256: (125.0, 0.50)},
    )
    candidates = list(range(448))

    assert lm.get_probs(candidates) == candidates

    assert [len(batch) for _, batch in calls] == [64, 128, 256]
    assert lm.effective_gpu_batch_size == 256
    assert lm.batch_telemetry == {
        "configured_batch_size": "auto",
        "effective_batch_size": 256,
        "max_batch_size": 256,
        "tuning_state": "converged",
        "candidates_per_second": 125.0,
        "peak_memory_bytes": 256000,
        "free_memory_bytes": 500,
        "total_memory_bytes": 1000,
        "memory_headroom": 0.50,
        "oom_backoff_events": 0,
        "failed_upper_bound": None,
    }


def test_auto_stops_at_throughput_plateau(monkeypatch):
    lm = build_lm(monkeypatch)
    calls = install_recording_scores(
        lm, {64: (100.0, 0.80), 128: (104.9, 0.70)}
    )
    candidates = list(range(192))

    assert lm.get_probs(candidates) == candidates

    assert [len(batch) for _, batch in calls] == [64, 128]
    assert lm.effective_gpu_batch_size == 64
    assert lm.batch_telemetry["tuning_state"] == "converged"


def test_auto_stops_before_growth_when_baseline_memory_is_low(monkeypatch):
    lm = build_lm(monkeypatch)
    calls = install_recording_scores(lm, {64: (100.0, 0.19)})
    candidates = list(range(192))

    assert lm.get_probs(candidates) == candidates

    assert [len(batch) for _, batch in calls] == [64, 64, 64]
    assert [kind for kind, _ in calls] == ["timed", "ordinary", "ordinary"]
    assert lm.effective_gpu_batch_size == 64
    assert lm.batch_telemetry["memory_headroom"] == 0.19


def test_auto_rejects_growth_that_leaves_too_little_memory(monkeypatch):
    lm = build_lm(monkeypatch)
    install_recording_scores(
        lm, {64: (100.0, 0.80), 128: (120.0, 0.19)}
    )

    lm.get_probs(list(range(192)))

    assert lm.effective_gpu_batch_size == 64
    assert lm.batch_telemetry["tuning_state"] == "converged"
    assert lm.batch_telemetry["memory_headroom"] == 0.19


def test_small_call_and_tail_do_not_trigger_tuning_measurements(monkeypatch):
    lm = build_lm(monkeypatch)
    calls = install_recording_scores(lm)
    candidates = list(range(63))

    assert lm.get_probs(candidates) == candidates

    assert calls == [("ordinary", candidates)]
    assert lm.batch_telemetry["tuning_state"] == "warming"
    assert lm.batch_telemetry["candidates_per_second"] is None


def test_partial_growth_batch_remains_an_untimed_tail(monkeypatch):
    lm = build_lm(monkeypatch)
    calls = install_recording_scores(lm, {64: (100.0, 0.80)})
    candidates = list(range(191))

    assert lm.get_probs(candidates) == candidates

    assert [(kind, len(batch)) for kind, batch in calls] == [
        ("timed", 64),
        ("ordinary", 64),
        ("ordinary", 63),
    ]
    assert lm.effective_gpu_batch_size == 64
    assert lm.batch_telemetry["tuning_state"] == "tuning"


def test_cuda_oom_retries_exact_slice_without_reordering(monkeypatch):
    lm = build_lm(monkeypatch, max_batch_size=128)
    calls = []
    growth_attempts = 0
    oom_type = minicons_lm.torch.cuda.OutOfMemoryError
    monkeypatch.setattr(minicons_lm.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(minicons_lm.gc, "collect", lambda: 0)

    def score(batch):
        calls.append(("ordinary", list(batch)))
        return list(batch)

    def score_timed(batch):
        nonlocal growth_attempts
        calls.append(("timed", list(batch)))
        if len(batch) == 128:
            growth_attempts += 1
            raise oom_type("synthetic CUDA OOM")
        return list(batch), {
            "throughput": 100.0,
            "peak_memory": 1000,
            "memory_headroom": 0.80,
        }

    lm.get_batch_scores = score
    lm._score_timed_batch = score_timed
    recover = lm._recover_from_oom
    cleanup_exceptions = []

    def recover_without_live_exception(batch_size):
        cleanup_exceptions.append(sys.exc_info()[1])
        recover(batch_size)

    lm._recover_from_oom = recover_without_live_exception
    candidates = list(range(192))

    assert lm.get_probs(candidates) == candidates

    assert growth_attempts == 1
    assert calls == [
        ("timed", candidates[:64]),
        ("timed", candidates[64:192]),
        ("ordinary", candidates[64:128]),
        ("ordinary", candidates[128:192]),
    ]
    assert lm.effective_gpu_batch_size == 64
    assert lm.batch_telemetry["failed_upper_bound"] == 128
    assert lm.batch_telemetry["oom_backoff_events"] == 1
    assert cleanup_exceptions == [None]


def test_converged_controller_still_recovers_from_later_oom(monkeypatch):
    lm = build_lm(monkeypatch, max_batch_size=128)
    lm.effective_gpu_batch_size = 128
    lm._tuning_state = "converged"
    lm._last_known_safe_batch_size = 128
    attempts = []
    oom_type = minicons_lm.torch.cuda.OutOfMemoryError
    monkeypatch.setattr(minicons_lm.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(minicons_lm.gc, "collect", lambda: 0)

    def score(batch):
        attempts.append(list(batch))
        if len(attempts) == 1:
            raise oom_type("synthetic CUDA OOM")
        return list(batch)

    lm.get_batch_scores = score
    candidates = list(range(128))

    assert lm.get_probs(candidates) == candidates

    assert attempts == [candidates, candidates[:64], candidates[64:]]
    assert lm.effective_gpu_batch_size == 64
    assert lm.batch_telemetry["oom_backoff_events"] == 1


def test_single_candidate_cuda_oom_is_reraised(monkeypatch):
    lm = build_lm(monkeypatch, max_batch_size=1)
    oom = minicons_lm.torch.cuda.OutOfMemoryError("synthetic CUDA OOM")
    lm._score_timed_batch = lambda _batch: (_ for _ in ()).throw(oom)

    with pytest.raises(minicons_lm.torch.cuda.OutOfMemoryError) as caught:
        lm.get_probs(["candidate"])

    assert caught.value is oom
    assert lm.batch_telemetry["oom_backoff_events"] == 0


def test_non_cuda_runtime_error_is_not_treated_as_oom(monkeypatch):
    lm = build_lm(monkeypatch)
    error = RuntimeError("model failure")
    lm._score_timed_batch = lambda _batch: (_ for _ in ()).throw(error)

    with pytest.raises(RuntimeError) as caught:
        lm.get_probs(list(range(64)))

    assert caught.value is error
    assert lm.batch_telemetry["oom_backoff_events"] == 0


def test_explicit_integer_mode_keeps_fixed_microbatches(monkeypatch):
    lm = build_lm(monkeypatch, batch_size=3, max_batch_size=2)
    calls = install_recording_scores(lm)
    candidates = list(range(8))

    assert lm.get_probs(candidates) == candidates

    assert [len(batch) for _, batch in calls] == [3, 3, 2]
    assert lm.effective_gpu_batch_size == 3
    assert lm.batch_telemetry["tuning_state"] == "fixed"


def test_auto_cpu_mode_uses_safe_fixed_size_without_cuda_timing(monkeypatch):
    lm = build_lm(monkeypatch, device="cpu")
    calls = install_recording_scores(lm)
    candidates = list(range(130))

    assert lm.get_probs(candidates) == candidates

    assert [len(batch) for _, batch in calls] == [64, 64, 2]
    assert lm.batch_telemetry["tuning_state"] == "cpu"


def test_two_scorers_keep_independent_adaptive_state(monkeypatch):
    segmenter = build_lm(monkeypatch, max_batch_size=128)
    reranker = build_lm(monkeypatch, max_batch_size=128)
    install_recording_scores(
        segmenter, {64: (100.0, 0.80), 128: (110.0, 0.60)}
    )
    install_recording_scores(
        reranker, {64: (100.0, 0.80), 128: (101.0, 0.60)}
    )
    candidates = list(range(192))

    segmenter.get_probs(candidates)
    reranker.get_probs(candidates)

    assert segmenter.effective_gpu_batch_size == 128
    assert reranker.effective_gpu_batch_size == 64


def test_timed_batch_synchronizes_cuda_and_reports_memory(monkeypatch):
    lm = build_lm(monkeypatch)
    lm.get_batch_scores = lambda batch: list(batch)
    events = []
    synchronized = []

    class FakeEvent:
        def __init__(self, enable_timing):
            assert enable_timing is True
            events.append(self)

        def record(self):
            self.recorded = True

        def elapsed_time(self, other):
            assert self.recorded and other.recorded
            return 20.0

    monkeypatch.setattr(minicons_lm.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        minicons_lm.torch.cuda, "device", lambda device: nullcontext()
    )
    monkeypatch.setattr(
        minicons_lm.torch.cuda, "reset_peak_memory_stats", lambda device: None
    )
    monkeypatch.setattr(
        minicons_lm.torch.cuda,
        "synchronize",
        lambda device: synchronized.append(device),
    )
    monkeypatch.setattr(
        minicons_lm.torch.cuda, "mem_get_info", lambda device: (600, 1000)
    )
    monkeypatch.setattr(
        minicons_lm.torch.cuda, "max_memory_allocated", lambda device: 350
    )

    scores, metrics = lm._score_timed_batch(list(range(10)))

    assert scores == list(range(10))
    assert len(events) == 2
    assert synchronized == ["cuda"]
    assert metrics == {
        "throughput": 500.0,
        "peak_memory": 350,
        "free_memory": 600,
        "total_memory": 1000,
        "memory_headroom": 0.6,
    }


@pytest.mark.parametrize("value", [0, -1, True, "AUTO", "64"])
def test_invalid_batch_size_is_rejected(value):
    with pytest.raises(ValueError):
        minicons_lm.validate_batch_size(value)


@pytest.mark.parametrize("value", [0, -1, True, "512"])
def test_invalid_max_batch_size_is_rejected(value):
    with pytest.raises(ValueError):
        minicons_lm.validate_max_batch_size(value)
