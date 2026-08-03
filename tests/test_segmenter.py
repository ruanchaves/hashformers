from pathlib import Path

import pytest
import torch

from hashformers.beamsearch.algorithm import Beamsearch
from hashformers.beamsearch.reranker import Reranker
from hashformers.ensemble.top2_fusion import Top2_Ensembler
from hashformers.segmenter.segmenter import BaseWordSegmenter


TEST_DATA_DIR = Path(__file__).parent.absolute()
CUDA_IS_AVAILABLE = torch.cuda.is_available()

if not CUDA_IS_AVAILABLE:
    raise Exception("A GPU is required for these tests.")


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="A GPU is not available.")
def test_cuda_availability():
    """Verify that the GPU-only segmenter tests run with CUDA available."""
    assert CUDA_IS_AVAILABLE


@pytest.fixture(scope="module")
def word_segmenter_gpt2_bert():
    """Build a GPU-backed segmenter, reranker, and ensemble pipeline."""
    segmenter = Beamsearch(
        model_name_or_path="distilgpt2",
        gpu_batch_size=1000,
    )
    reranker = Reranker(
        model_name_or_path="bert-base-cased",
        gpu_batch_size=1000,
    )
    ensembler = Top2_Ensembler()
    return BaseWordSegmenter(
        segmenter=segmenter,
        reranker=reranker,
        ensembler=ensembler,
    )


SEGMENTER_FIXTURES = [pytest.lazy_fixture("word_segmenter_gpt2_bert")]


@pytest.mark.parametrize("word_segmenter", SEGMENTER_FIXTURES)
def test_word_segmenter_output(word_segmenter):
    """Verify selected segmentations preserve every input character."""
    hashtags = [
        "minecraf",
        "ourmomentfragrance",
        "waybackwhen",
    ]

    predictions = word_segmenter.predict(hashtags).output
    prediction_characters = [value.replace(" ", "") for value in predictions]

    assert prediction_characters == hashtags
