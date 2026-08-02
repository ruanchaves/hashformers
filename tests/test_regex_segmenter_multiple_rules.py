from hashformers import RegexWordSegmenter, TweetSegmenter


def test_default_rule_preserves_single_rule_behavior():
    """Verify that the default camel-case segmentation remains unchanged.

    The multiple-rule fix must preserve existing behavior for the default rule.
    """
    segmenter = RegexWordSegmenter()

    assert segmenter.segment(["UnaGenialidad"]) == ["Una Genialidad"]


def test_multiple_rules_are_applied_sequentially_per_input():
    """Verify that every rule transforms each input before it is emitted.

    Two rules applied to two inputs must produce two outputs, in input order.
    """
    segmenter = RegexWordSegmenter(
        regex_rules=[r"([A-Z]+)", r"([0-9]+)"]
    )

    assert segmenter.segment(["fooBAR123", "zipZIP456"]) == [
        "foo BAR 123",
        "zip ZIP 456",
    ]


def test_tweet_segmenter_receives_one_segmentation_per_hashtag():
    """Verify that multiple rules keep tweet hashtag replacements aligned.

    Each unique hashtag must receive exactly one fully transformed segmentation.
    """
    matcher = lambda tweets: [["fooBAR123"], ["zipZIP456"]]
    word_segmenter = RegexWordSegmenter(
        regex_rules=[r"([A-Z]+)", r"([0-9]+)"]
    )
    segmenter = TweetSegmenter(matcher=matcher, word_segmenter=word_segmenter)

    result = segmenter.segment(["First #fooBAR123", "Second #zipZIP456"])

    assert result.output == ["First foo BAR 123", "Second zip ZIP 456"]
    assert len(result.word_segmenter_output.output) == 2
