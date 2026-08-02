from unittest.mock import Mock

import pandas as pd

from hashformers.beamsearch.data_structures import ProbabilityDictionary
from hashformers.segmenter.segmenter import BaseWordSegmenter


def _scores(preferred_segmentation):
    alternatives = {
        "foo bar": 1.0,
        "fo obar": 1.0,
    }
    alternatives[preferred_segmentation] = 0.0
    return ProbabilityDictionary(alternatives)


def test_reranker_output_is_used_when_ensemble_is_disabled():
    segmenter_run = _scores("foo bar")
    reranker_run = _scores("fo obar")
    segmenter_model = Mock()
    segmenter_model.run.return_value = segmenter_run
    reranker_model = Mock()
    reranker_model.rerank.return_value = reranker_run
    ensembler = Mock()
    segmenter = BaseWordSegmenter(segmenter_model, reranker_model, ensembler)

    result = segmenter.segment(["foobar"], use_ensembler=False)

    assert result == ["fo obar"]
    reranker_model.rerank.assert_called_once_with(segmenter_run)
    ensembler.run.assert_not_called()


def test_reranker_output_is_used_when_no_ensembler_is_configured():
    segmenter_run = _scores("foo bar")
    reranker_run = _scores("fo obar")
    segmenter_model = Mock()
    segmenter_model.run.return_value = segmenter_run
    reranker_model = Mock()
    reranker_model.rerank.return_value = reranker_run
    segmenter = BaseWordSegmenter(segmenter_model, reranker_model)

    result = segmenter.segment(["foobar"])

    assert result == ["fo obar"]
    reranker_model.rerank.assert_called_once_with(segmenter_run)


def test_ensemble_output_still_takes_precedence():
    segmenter_run = _scores("foo bar")
    reranker_run = _scores("fo obar")
    ensemble_run = _scores("foo bar")
    segmenter_model = Mock()
    segmenter_model.run.return_value = segmenter_run
    reranker_model = Mock()
    reranker_model.rerank.return_value = reranker_run
    ensembler = Mock()
    ensembler.run.return_value = ensemble_run
    segmenter = BaseWordSegmenter(segmenter_model, reranker_model, ensembler)

    result = segmenter.segment(["foobar"])

    assert result == ["foo bar"]
    reranker_model.rerank.assert_called_once_with(segmenter_run)
    ensembler.run.assert_called_once_with(segmenter_run, reranker_run)


def test_segmenter_output_is_used_when_reranking_is_disabled():
    segmenter_run = _scores("foo bar")
    segmenter_model = Mock()
    segmenter_model.run.return_value = segmenter_run
    reranker_model = Mock()
    segmenter = BaseWordSegmenter(segmenter_model, reranker_model)

    result = segmenter.segment(["foobar"], use_reranker=False)

    assert result == ["foo bar"]
    reranker_model.rerank.assert_not_called()


def test_precomputed_probability_dictionary_skips_segmenter_model():
    precomputed_run = _scores("foo bar")
    segmenter_model = Mock()
    segmenter = BaseWordSegmenter(segmenter_model)

    result = segmenter.segment(
        ["foobar"],
        segmenter_run=precomputed_run,
        use_reranker=False,
    )

    assert result == ["foo bar"]
    segmenter_model.run.assert_not_called()


def test_precomputed_dataframe_still_skips_segmenter_model():
    precomputed_run = pd.DataFrame(
        [
            {"segmentation": "foo bar", "score": 0.0},
            {"segmentation": "fo obar", "score": 1.0},
        ]
    )
    segmenter_model = Mock()
    segmenter = BaseWordSegmenter(segmenter_model)

    result = segmenter.segment(
        ["foobar"],
        segmenter_run=precomputed_run,
        use_reranker=False,
    )

    assert result == ["foo bar"]
    segmenter_model.run.assert_not_called()


def test_omitted_segmenter_run_still_invokes_segmenter_model():
    model_run = _scores("foo bar")
    segmenter_model = Mock()
    segmenter_model.run.return_value = model_run
    segmenter = BaseWordSegmenter(segmenter_model)

    result = segmenter.segment(["foobar"], use_reranker=False)

    assert result == ["foo bar"]
    segmenter_model.run.assert_called_once_with(["foobar"])
