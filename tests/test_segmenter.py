import pytest
import torch
import hashformers
from hashformers import (RegexWordSegmenter, TweetSegmenter,
                                   TwitterTextMatcher)
from hashformers.beamsearch.algorithm import Beamsearch
from hashformers.beamsearch.reranker import Reranker
from hashformers.ensemble.top2_fusion import Top2_Ensembler
from hashformers.segmenter.segmenter import BaseWordSegmenter
from pathlib import Path
import dataclasses 

TEST_DATA_DIR = Path(__file__).parent.absolute()
CUDA_IS_AVAILABLE = torch.cuda.is_available()

if not CUDA_IS_AVAILABLE:
    raise Exception("A GPU is required for these tests.")

@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="A GPU is not available.")
def test_cuda_availability():
    """
    Checks if CUDA is available for the running tests.
    
    Raises:
        Exception: If CUDA is not available.
    """
    assert CUDA_IS_AVAILABLE

@pytest.fixture(scope="module")
def tweet_segmenter():
    """
    Initializes and returns a TweetSegmenter object with a TwitterTextMatcher and a RegexWordSegmenter.
    
    Returns:
        TweetSegmenter: An instance of the TweetSegmenter class.
    """
    return TweetSegmenter(
        matcher=TwitterTextMatcher(),
        word_segmenter=RegexWordSegmenter()
    )

@pytest.fixture(scope="module")
def word_segmenter_gpt2_bert():
    """
    Initializes and returns a BaseWordSegmenter object with Beamsearch, Reranker, and Top2_Ensembler.
    
    Returns:
        BaseWordSegmenter: An instance of the BaseWordSegmenter class.
    """

    segmenter = Beamsearch(
        model_name_or_path="distilgpt2",
        gpu_batch_size=1000
    )

    reranker = Reranker(
        model_name_or_path="bert-base-cased",
        gpu_batch_size=1000
    )

    ensembler = Top2_Ensembler()

    ws = BaseWordSegmenter(
        segmenter=segmenter,
        reranker=reranker,
        ensembler=ensembler
    )
    return ws

SEGMENTER_FIXTURES = [
    pytest.lazy_fixture("word_segmenter_gpt2_bert")
]

@pytest.mark.parametrize('word_segmenter', SEGMENTER_FIXTURES)
def test_word_segmenter_output(word_segmenter):
    """
    Tests the predict function of the provided word_segmenter.
    
    Args:
        word_segmenter (BaseWordSegmenter): The word_segmenter to be tested.
    """
    test_boun_hashtags = [
        "minecraf",
        "ourmomentfragrance",
        "waybackwhen"
    ]

    predictions = word_segmenter.predict(test_boun_hashtags).output
    predictions_chars = [ x.replace(" ", "") for x in predictions ]

    assert all([
        predictions_chars[0] == "minecraf",
        predictions_chars[1] == "ourmomentfragrance",
        predictions_chars[2] == "waybackwhen"
    ])

def test_twitter_text_matcher():
    """
    Tests the functionality of the TwitterTextMatcher.
    """
    matcher = TwitterTextMatcher()
    result = matcher(["esto es #UnaGenialidad"])

    assert result == [["UnaGenialidad"]]

def test_regex_word_segmentation():
    """
    Tests the predict function of the RegexWordSegmenter.
    """
    ws = RegexWordSegmenter()
    test_case = ["UnaGenialidad"]
    prediction = ws.predict(test_case)

    assert prediction.output == ["Una Genialidad"]

def test_hashtag_container(tweet_segmenter):
    """
    Tests the build_hashtag_container method of the provided tweet_segmenter.
    
    Args:
        tweet_segmenter (TweetSegmenter): The tweet_segmenter to be tested.
    """
    original_tweet = "esto es #UnaGenialidad"
    hashtag_container, word_segmenter_output = tweet_segmenter.build_hashtag_container([original_tweet])

    assert all([
        hashtag_container.hashtags == [['UnaGenialidad']],
        hashtag_container.hashtag_set == ['UnaGenialidad'],
        hashtag_container.replacement_dict == {'#UnaGenialidad': 'Una Genialidad'},
        isinstance(word_segmenter_output, hashformers.segmenter.WordSegmenterOutput)
    ])

def test_tweet_segmentation(tweet_segmenter):
    """
    Tests the segmentation process of the provided tweet_segmenter.
    
    Args:
        tweet_segmenter (TweetSegmenter): The tweet_segmenter to be tested.
    """
    original_tweet = "esto es #UnaGenialidad"
    expected_tweet = "esto es Una Genialidad"
    hashtag_container, word_segmenter_output = tweet_segmenter.build_hashtag_container([original_tweet])
    tweet = list(tweet_segmenter.segmented_tweet_generator([original_tweet], *dataclasses.astuple(hashtag_container), flag=0))[0]
    
    assert tweet == expected_tweet

def test_tweet_segmenter_output_format(tweet_segmenter):
    """
    Tests the predict method's output of the provided tweet_segmenter.
    
    Args:
        tweet_segmenter (TweetSegmenter): The tweet_segmenter to be tested.
    """
    original_tweet = "esto es #UnaGenialidad"
    expected_tweet = "esto es Una Genialidad"

    output_tweets = tweet_segmenter.predict([original_tweet])
    output_tweets = output_tweets.output
    
    assert output_tweets[0] == expected_tweet