import inspect

from hashformers.beamsearch.algorithm import Beamsearch
from hashformers.beamsearch.reranker import Reranker
from hashformers.segmenter import auto as auto_segmenter
from hashformers.segmenter.auto import TransformerWordSegmenter


class RecordingModel:
    def __init__(self):
        self.calls = []

    def get_probs(self, candidates):
        self.calls.append(candidates)
        return list(range(len(candidates)))


def test_candidates_are_deduplicated_before_scoring():
    beamsearch = object.__new__(Beamsearch)
    beamsearch.model = RecordingModel()
    probabilities = {"already scored": 9}

    result = beamsearch.update_probabilities(
        ["first", "duplicate", "already scored", "duplicate", "second"],
        probabilities,
    )

    assert beamsearch.model.calls == [["first", "duplicate", "second"]]
    assert result == {
        "already scored": 9,
        "first": 0,
        "duplicate": 1,
        "second": 2,
    }


def test_empty_pending_batch_does_not_call_scorer():
    beamsearch = object.__new__(Beamsearch)
    beamsearch.model = RecordingModel()

    beamsearch.update_probabilities(["cached", "cached"], {"cached": 1})

    assert beamsearch.model.calls == []


def test_beamsearch_leaves_auto_batching_to_the_scorer():
    beamsearch = object.__new__(Beamsearch)
    beamsearch.gpu_batch_size = "auto"
    beamsearch.model = RecordingModel()
    beamsearch.next_step = lambda _tree: ["first", "second"]
    beamsearch.trim_tree = lambda tree, _probabilities, _topk: tree

    result = beamsearch.run(["source"], steps=1)

    assert beamsearch.model.calls == [["first", "second"]]
    assert result.dictionary == {"first": 0, "second": 1}


def test_python_api_forwards_independent_auto_batch_limits(monkeypatch):
    calls = {}

    class RecordingBeamsearch:
        def __init__(self, **kwargs):
            calls["segmenter"] = kwargs

    class RecordingReranker:
        def __init__(self, **kwargs):
            calls["reranker"] = kwargs

    monkeypatch.setattr(auto_segmenter, "Beamsearch", RecordingBeamsearch)
    monkeypatch.setattr(auto_segmenter, "Reranker", RecordingReranker)

    TransformerWordSegmenter(
        segmenter_gpu_batch_size="auto",
        segmenter_max_gpu_batch_size=256,
        reranker_model_name_or_path="reranker",
        reranker_gpu_batch_size="auto",
        reranker_max_gpu_batch_size=128,
    )

    assert calls["segmenter"]["gpu_batch_size"] == "auto"
    assert calls["segmenter"]["max_gpu_batch_size"] == 256
    assert calls["reranker"]["gpu_batch_size"] == "auto"
    assert calls["reranker"]["max_gpu_batch_size"] == 128


def test_performance_defaults():
    beamsearch_init = inspect.signature(Beamsearch.__init__).parameters
    reranker_init = inspect.signature(Reranker.__init__).parameters
    segmenter_init = inspect.signature(TransformerWordSegmenter.__init__).parameters
    segment = inspect.signature(TransformerWordSegmenter.segment).parameters

    assert beamsearch_init["gpu_batch_size"].default == 64
    assert reranker_init["gpu_batch_size"].default == 64
    assert segmenter_init["segmenter_gpu_batch_size"].default == 64
    assert segmenter_init["reranker_gpu_batch_size"].default == 64
    assert segmenter_init["segmenter_max_gpu_batch_size"].default == 512
    assert segmenter_init["reranker_max_gpu_batch_size"].default == 512
    assert segment["topk"].default == 5
    assert segment["steps"].default == 5
