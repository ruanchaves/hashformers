import dataclasses

import hashformers
import pytest
from hashformers import RegexWordSegmenter, TweetSegmenter, TwitterTextMatcher
from hashformers.segmenter import auto as auto_module


@pytest.fixture(scope="module")
def tweet_segmenter():
    return TweetSegmenter(
        matcher=TwitterTextMatcher(),
        word_segmenter=RegexWordSegmenter(),
    )


def test_twitter_text_matcher():
    matcher = TwitterTextMatcher()
    result = matcher(["esto es #UnaGenialidad"])

    assert result == [["UnaGenialidad"]]


def test_regex_word_segmentation():
    ws = RegexWordSegmenter()
    prediction = ws.predict(["UnaGenialidad"])

    assert prediction.output == ["Una Genialidad"]


def test_hashtag_container(tweet_segmenter):
    original_tweet = "esto es #UnaGenialidad"
    hashtag_container, word_segmenter_output = tweet_segmenter.build_hashtag_container([original_tweet])

    assert all(
        [
            hashtag_container.hashtags == [["UnaGenialidad"]],
            hashtag_container.hashtag_set == ["UnaGenialidad"],
            hashtag_container.replacement_dict == {"#UnaGenialidad": "Una Genialidad"},
            isinstance(word_segmenter_output, hashformers.segmenter.WordSegmenterOutput),
        ]
    )


def test_tweet_segmentation(tweet_segmenter):
    original_tweet = "esto es #UnaGenialidad"
    expected_tweet = "esto es Una Genialidad"
    hashtag_container, _ = tweet_segmenter.build_hashtag_container([original_tweet])
    tweet = list(
        tweet_segmenter.segmented_tweet_generator(
            [original_tweet],
            *dataclasses.astuple(hashtag_container),
            flag=0,
        )
    )[0]

    assert tweet == expected_tweet


def test_tweet_segmenter_output_format(tweet_segmenter):
    original_tweet = "esto es #UnaGenialidad"
    expected_tweet = "esto es Una Genialidad"

    output_tweets = tweet_segmenter.predict([original_tweet]).output

    assert output_tweets[0] == expected_tweet


def test_transformer_word_segmenter_uses_canonical_defaults(monkeypatch):
    recorded = {}

    class FakeBeamsearch:
        def __init__(self, **kwargs):
            recorded["segmenter"] = kwargs

    class FakeReranker:
        def __init__(self, **kwargs):
            recorded["reranker"] = kwargs

    class FakeEnsembler:
        pass

    monkeypatch.setattr(auto_module, "Beamsearch", FakeBeamsearch)
    monkeypatch.setattr(auto_module, "Reranker", FakeReranker)
    monkeypatch.setattr(auto_module, "Top2_Ensembler", FakeEnsembler)

    auto_module.TransformerWordSegmenter(
        segmenter_model_name_or_path="segmenter-demo",
        reranker_model_name_or_path="reranker-demo",
    )

    assert recorded["segmenter"]["model_type"] == "incremental"
    assert recorded["reranker"]["model_type"] == "masked"
