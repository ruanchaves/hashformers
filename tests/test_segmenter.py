from hashformers.segmenter.unigram_segmenter import UnigramWordSegmenter
import pytest
import torch

import hashformers

from hashformers import (RegexWordSegmenter, TweetSegmenter,
                                   TwitterTextMatcher)
from hashformers.beamsearch.algorithm import Beamsearch
from hashformers.beamsearch.reranker import Reranker
from hashformers.ensemble.top2_fusion import Top2_Ensembler
from hashformers.segmenter.segmenter import BaseWordSegmenter
from hashformers.segmenter.segmenter import WordSegmenterCascade
from hashformers.segmenter.data_structures import CascadeNode
import dataclasses
from pathlib import Path

test_data_dir = Path(__file__).parent.absolute()

cuda_is_available = torch.cuda.is_available()

@pytest.mark.skipif(not cuda_is_available, reason="A GPU is not available.")
def test_cuda():
    assert torch.cuda.is_available() == True

@pytest.fixture(scope="module")
def tweet_segmenter():
    return TweetSegmenter(
        matcher=TwitterTextMatcher(),
        word_segmenter=RegexWordSegmenter()
    )

@pytest.fixture(scope="module")
def word_segmenter_gpt2_bert():

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

@pytest.fixture(scope="module")
def word_segmenter_unigram():
    segmenter = UnigramWordSegmenter(
        max_split_length=20
    )
    ws = BaseWordSegmenter(
        segmenter=segmenter,
        reranker=None,
        ensembler=None
    )
    return ws

@pytest.fixture(scope="module")
def word_segmenter_unigram_bert():
    segmenter = UnigramWordSegmenter(
        max_split_length=20
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

@pytest.fixture(scope="module")
def word_segmenter_unigram_gpt2():
    segmenter = UnigramWordSegmenter(
        max_split_length=20
    )

    reranker = Reranker(
        model_name_or_path="distilgpt2",
        model_type="gpt2",
        gpu_batch_size=1000
    )

    ensembler = Top2_Ensembler()

    ws = BaseWordSegmenter(
        segmenter=segmenter,
        reranker=reranker,
        ensembler=ensembler
    )
    return ws

@pytest.fixture(scope="module")
def word_segmenter_unigram_gpt2_bert_cascade():
    
    segmenter = UnigramWordSegmenter(
        max_split_length=20
    )

    reranker_1 = Reranker(
        model_name_or_path="distilgpt2",
        model_type="gpt2",
        gpu_batch_size=1000
    )

    reranker_2 = Reranker(
        model_name_or_path="bert-base-cased",
        gpu_batch_size=1000
    )

    ensembler = Top2_Ensembler()

    ws1 = BaseWordSegmenter(
            segmenter=segmenter,
            reranker=reranker_1,
            ensembler=ensembler
    )

    ws2 = BaseWordSegmenter(
            segmenter=None,
            reranker=reranker_2,
            ensembler=ensembler
    )


    node1 = CascadeNode(
        word_segmenter=ws1,
        word_segmenter_kwargs={
            "segmenter_kwargs": {
                "topk": 5,
                "steps": 5
            },
            "reranker_kwargs": {

            },
            "ensembler_kwargs": {
                "alpha": 0.222,
                "beta": 0.111
            },
            "use_reranker": True,
            "use_ensembler": True
        }
    )

    node2 = CascadeNode(
        word_segmenter=ws2,
        word_segmenter_kwargs={
            "segmenter_kwargs": {

            },
            "reranker_kwargs": {

            },
            "ensembler_kwargs": {
                "alpha": 0.222,
                "beta": 0.111
            },
            "use_reranker": True,
            "use_ensembler": True
        }
    )

    cascade = [node1, node2]

    return WordSegmenterCascade(cascade)

if cuda_is_available:
    segmenter_fixtures = [
        "word_segmenter_gpt2_bert", 
        "word_segmenter_unigram",
        "word_segmenter_unigram_bert",
        "word_segmenter_unigram_gpt2",
        "word_segmenter_unigram_gpt2_bert_cascade"
    ]
else:
    segmenter_fixtures = ["word_segmenter_unigram"]

segmenter_fixtures = [
    pytest.lazy_fixture(x) for x in segmenter_fixtures
]

@pytest.mark.parametrize('word_segmenter', segmenter_fixtures)
def test_word_segmenter_output_format(word_segmenter):
    test_boun_hashtags = [
        "minecraf",
        "ourmomentfragrance",
        "waybackwhen"
    ]

    predictions = word_segmenter.predict(test_boun_hashtags).output

    predictions_chars = [ x.replace(" ", "") for x in predictions ]
    
    assert predictions_chars[0] == "minecraf"
    assert predictions_chars[1] == "ourmomentfragrance"
    assert predictions_chars[2] == "waybackwhen"

def test_matcher():
    matcher = TwitterTextMatcher()
    result = matcher(["esto es #UnaGenialidad"])
    
    assert result == [["UnaGenialidad"]]

def test_regex_word_segmenter():
    ws = RegexWordSegmenter()
    test_case = ["UnaGenialidad"]
    prediction = ws.predict(test_case)
    
    assert prediction.output == ["Una Genialidad"]

def test_hashtag_container(tweet_segmenter):
    original_tweet = "esto es #UnaGenialidad"
    hashtag_container, word_segmenter_output = tweet_segmenter.build_hashtag_container([original_tweet])

    assert hashtag_container.hashtags == [['UnaGenialidad']]
    assert hashtag_container.hashtag_set == ['UnaGenialidad']
    assert hashtag_container.replacement_dict == {'#UnaGenialidad': 'Una Genialidad'}
    assert isinstance(word_segmenter_output, hashformers.segmenter.WordSegmenterOutput)

def test_tweet_segmenter_generator(tweet_segmenter):
    original_tweet = "esto es #UnaGenialidad"
    expected_tweet = "esto es Una Genialidad"
    hashtag_container, word_segmenter_output = tweet_segmenter.build_hashtag_container([original_tweet])
    tweet = list(tweet_segmenter.segmented_tweet_generator([original_tweet], *dataclasses.astuple(hashtag_container), flag=0))[0]
    
    assert tweet == expected_tweet

def test_tweet_segmenter_output_format(tweet_segmenter):

    original_tweet = "esto es #UnaGenialidad"
    expected_tweet = "esto es Una Genialidad"

    output_tweets = tweet_segmenter.predict([original_tweet])
    output_tweets = output_tweets.output
    
    assert output_tweets[0] == expected_tweet