from hashformers.beamsearch.algorithm import Beamsearch


def build_beamsearch():
    """Build a beam search instance without loading a language model.

    Returns:
        Beamsearch: An uninitialized instance suitable for testing ``trim_tree``.
    """
    return object.__new__(Beamsearch)


def test_duplicate_hypotheses_do_not_consume_beam_slots():
    beamsearch = build_beamsearch()
    tree = ["a b cd", "a b cd", "ab cd", "abc d"]
    scores = {"a b cd": 1, "ab cd": 2, "abc d": 3}

    result = beamsearch.trim_tree(tree, scores, topk=2)

    assert result == ["a b cd", "ab cd"]


def test_trim_tree_preserves_input_order_when_scores_are_tied():
    beamsearch = build_beamsearch()
    tree = ["ab cd", "a bcd", "abc d"]
    scores = {"ab cd": 2, "a bcd": 1, "abc d": 1}

    result = beamsearch.trim_tree(tree, scores, topk=2)

    assert result == ["a bcd", "abc d"]


def test_trim_tree_deduplicates_character_groups_independently():
    beamsearch = build_beamsearch()
    tree = [
        "a b cd",
        "a b cd",
        "ab cd",
        "w x yz",
        "w x yz",
        "wx yz",
    ]
    scores = {
        "a b cd": 1,
        "ab cd": 2,
        "w x yz": 4,
        "wx yz": 3,
    }

    result = beamsearch.trim_tree(tree, scores, topk=2)

    assert result == ["a b cd", "ab cd", "wx yz", "w x yz"]
