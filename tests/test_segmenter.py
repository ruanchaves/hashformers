import pytest

import dataclasses
import json
import os
from pathlib import Path

import hashformers
import torch
from hashformers import (RegexWordSegmenter, TweetSegmenter,
                                   TwitterTextMatcher, prune_segmenter_layers)

test_data_dir = Path(__file__).parent.absolute()
cuda_is_available = torch.cuda.is_available()

with open(os.path.join(test_data_dir,"fixtures/word_segmenters.json"), "r") as f:
    word_segmenter_params = json.load(f)
    word_segmenter_test_ids = []
    for row in word_segmenter_params:
        class_name = row["class"]
        segmenter = row["init_kwargs"].get("segmenter_model_name_or_path", "O")
        reranker = row["init_kwargs"].get("reranker_model_name_or_path", "O")
        id_string = "{0}_{1}_{2}".format(class_name, segmenter, reranker)
        word_segmenter_test_ids.append(id_string)

@pytest.fixture(scope="module")
def tweet_segmenter():
    return TweetSegmenter()

@pytest.fixture(scope="module", params=word_segmenter_params, ids=word_segmenter_test_ids)
def word_segmenter(request):
    
    word_segmenter_class = request.param["class"]
    word_segmenter_init_kwargs = request.param["init_kwargs"]
    word_segmenter_predict_kwargs = request.param["predict_kwargs"]

    WordSegmenterClass = getattr(hashformers, word_segmenter_class)

    class WordSegmenterClassWrapper(WordSegmenterClass):

        def __init__(self, **kwargs):
            return super().__init__(**kwargs)
        
        def predict(self, *args):
            return super().predict(*args, **word_segmenter_predict_kwargs)

    WordSegmenterClassWrapper.__name__ = request.param["class"] + "ClassWrapper"

    ws = WordSegmenterClassWrapper(**word_segmenter_init_kwargs)
    
    if request.param.get("prune", False):
        ws = prune_segmenter_layers(ws, layer_list=[0])

    return ws

@pytest.mark.skipif(not cuda_is_available, reason="A GPU is not available.")
def test_word_segmenter_output_format(word_segmenter):
    
    test_boun_hashtags = [
        "minecraf",
        "ourmomentfragrance",
        "waybackwhen"
    ]

    predictions = word_segmenter.predict(test_boun_hashtags).output

    predictions_chars = [ x.replace(" ", "") for x in predictions ]
    
    assert all([x == y for x,y in zip(test_boun_hashtags, predictions_chars)])

def test_matcher():
    matcher = TwitterTextMatcher()
    result = matcher(["esto es #UnaGenialidad"])
    assert result == [["UnaGenialidad"]]

def test_regex_word_segmenter():
    ws = RegexWordSegmenter()
    test_case = ["UnaGenialidad"]
    expected_output=["Una Genialidad"]
    prediction = ws.predict(test_case) 
    error_message = "{0} != {1}".format(prediction, str(test_case))
    assert prediction.output == expected_output, error_message

def test_hashtag_container(tweet_segmenter):
    original_tweets = [
        "esto es #UnaGenialidad"
    ]
    hashtag_container, word_segmenter_output = tweet_segmenter.build_hashtag_container(original_tweets)
    assert hashtag_container.hashtags == [['UnaGenialidad']]
    assert hashtag_container.hashtag_set == ['UnaGenialidad']
    assert hashtag_container.replacement_dict == {'#UnaGenialidad': 'Una Genialidad'}
    assert isinstance(word_segmenter_output, hashformers.segmenter.WordSegmenterOutput)

def test_tweet_segmenter_generator(tweet_segmenter):
    original_tweets = [
        "esto es #UnaGenialidad"
    ]
    hashtag_container, word_segmenter_output = tweet_segmenter.build_hashtag_container(original_tweets)
    for tweet in tweet_segmenter.segmented_tweet_generator(original_tweets, *dataclasses.astuple(hashtag_container), flag=0):
        assert tweet == "esto es Una Genialidad"

def test_tweet_segmenter_output_format(tweet_segmenter):

    original_tweets = [
        "esto es #UnaGenialidad"
    ]

    expected_tweets = [
        "esto es Una Genialidad"
    ]

    output_tweets = tweet_segmenter.predict(original_tweets)

    output_tweets = output_tweets.output

    assert type(output_tweets) == type([])

    assert len(original_tweets) == len(expected_tweets) == len(output_tweets)

    for idx, tweet in enumerate(original_tweets):
        assert expected_tweets[idx] == output_tweets[idx], \
            "{0} != {1}".format(expected_tweets[idx], output_tweets[idx])