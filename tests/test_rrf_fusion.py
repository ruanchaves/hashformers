import math

import pytest

from hashformers.beamsearch.data_structures import ProbabilityDictionary
from hashformers.ensemble.rrf_fusion import (
    ReciprocalRankFusionEnsembler,
    reciprocal_rank_fusion,
)
from hashformers.segmenter import BaseWordSegmenter


class _FixedSegmenter:
    def run(self, _word_list, **_kwargs):
        return ProbabilityDictionary({"abc": 1.0, "a bc": 2.0, "ab c": 3.0})


class _FixedReranker:
    def rerank(self, _segmenter_run, **_kwargs):
        return ProbabilityDictionary({"ab c": 1.0, "abc": 2.0, "a bc": 3.0})


def _model_free_segmenter(ensembler):
    return BaseWordSegmenter(
        segmenter=_FixedSegmenter(),
        reranker=_FixedReranker(),
        ensembler=ensembler,
    )


def test_known_rrf_scores_and_full_list_order():
    segmenter = ProbabilityDictionary({"ab": 1.0, "a b": 2.0, "a  b": 3.0})
    reranker = ProbabilityDictionary({"a  b": 10.0, "ab": 20.0, "a b": 30.0})

    result = reciprocal_rank_fusion([segmenter, reranker], rrf_k=60)

    assert list(result.dictionary) == ["ab", "a  b", "a b"]
    assert result.dictionary["ab"] == pytest.approx(-(1 / 61 + 1 / 62))
    assert len(result.dictionary) == 3


def test_rrf_can_select_candidate_outside_segmenter_top_two():
    segmenter = {"abc": 1.0, "a bc": 2.0, "ab c": 3.0}
    reranker = {"ab c": 1.0, "abc": 2.0, "a bc": 3.0}

    result = ReciprocalRankFusionEnsembler().run(
        segmenter,
        reranker,
        rrf_k=0,
        fusion_weights={"segmenter": 1.0, "reranker": 2.0},
    )

    assert next(iter(result.get_top_k(k=1))) == "ab c"


def test_base_segmenter_runs_weighted_rrf_and_exposes_every_ranking():
    segmenter = _model_free_segmenter(ensembler=object())

    output = segmenter.segment(
        ["abc"],
        fusion_method="rrf",
        ensembler_kwargs={
            "rrf_k": 0,
            "fusion_weights": {"segmenter": 1.0, "reranker": 2.0},
        },
        return_ranks=True,
    )

    assert output.output == ["ab c"]
    assert output.fusion_method == "rrf"
    assert output.segmenter_rank["segmentation"].tolist() == [
        "abc",
        "a bc",
        "ab c",
    ]
    assert output.reranker_rank["segmentation"].tolist() == [
        "ab c",
        "abc",
        "a bc",
    ]
    assert output.ensemble_rank["segmentation"].tolist() == [
        "ab c",
        "abc",
        "a bc",
    ]
    assert output.ensemble_rank["score"].tolist() == pytest.approx(
        [-7 / 3, -2.0, -7 / 6]
    )


def test_base_segmenter_preserves_top2_fusion_contract():
    class RecordingTop2:
        def __init__(self):
            self.call = None

        def run(self, segmenter_run, reranker_run, **kwargs):
            self.call = (segmenter_run, reranker_run, kwargs)
            return ProbabilityDictionary({"a bc": 0.0, "abc": 1.0})

    top2 = RecordingTop2()
    segmenter = _model_free_segmenter(ensembler=top2)

    output = segmenter.segment(
        ["abc"],
        fusion_method="top2",
        ensembler_kwargs={"alpha": 0.4, "beta": 0.6},
        return_ranks=True,
    )

    assert output.output == ["a bc"]
    assert output.fusion_method == "top2"
    assert top2.call[2] == {"alpha": 0.4, "beta": 0.6}


def test_rrf_requires_an_active_reranker_and_ensembler_before_segmentation():
    class UnusedSegmenter:
        def run(self, _word_list, **_kwargs):
            raise AssertionError("segmenter should not run")

    segmenter = BaseWordSegmenter(segmenter=UnusedSegmenter())

    with pytest.raises(
        ValueError,
        match="fusion_method=rrf requires an active reranker and ensembler",
    ):
        segmenter.segment(["abc"], fusion_method="rrf")


def test_rrf_depends_on_rank_not_score_magnitude():
    first = reciprocal_rank_fusion(
        [{"abc": 1.0, "a bc": 2.0}, {"a bc": 3.0, "abc": 4.0}]
    )
    scaled = reciprocal_rank_fusion(
        [{"abc": -1000.0, "a bc": 1e20}, {"a bc": -5.0, "abc": 99.0}]
    )
    assert list(first.dictionary) == list(scaled.dictionary)


def test_competition_ties_use_component_then_stable_order():
    result = reciprocal_rank_fusion(
        [{"abc": 1.0, "a bc": 1.0}, {"a bc": 2.0, "abc": 2.0}],
        rrf_k=0,
    )
    assert list(result.dictionary) == ["abc", "a bc"]


@pytest.mark.parametrize(
    ("rrf_k", "weights"),
    [
        (-1, [1.0, 1.0]),
        (math.inf, [1.0, 1.0]),
        (60, [-1.0, 1.0]),
        (60, [0.0, 0.0]),
        (60, [math.nan, 1.0]),
    ],
)
def test_invalid_options(rrf_k, weights):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([{"abc": 1.0}, {"abc": 2.0}], weights, rrf_k)


def test_union_includes_candidates_missing_from_a_component():
    result = reciprocal_rank_fusion(
        [{"abc": 1.0}, {"abc": 1.0, "a bc": 2.0}],
        rrf_k=0,
    )
    assert set(result.dictionary) == {"abc", "a bc"}
