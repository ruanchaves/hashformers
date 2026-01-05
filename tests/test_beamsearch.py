"""Tests for beamsearch algorithm module.

HASH-409: Improve Test Coverage - Phase 1.1
Tests for hashformers.beamsearch.algorithm
"""

import pytest
from hashformers.beamsearch.algorithm import (
    Beamsearch,
    has_consecutive_spaces_tokens,
    DOUBLE_SPACE_PATTERN
)
from hashformers.beamsearch.data_structures import Node, ProbabilityDictionary


class TestHasConsecutiveSpacesTokens:
    """Test the has_consecutive_spaces_tokens helper function."""

    def test_consecutive_spaces_at_start(self):
        """Consecutive space tokens at the start should return True."""
        token_ids = (3, 3, 1, 2, 4)  # 3 is space token
        assert has_consecutive_spaces_tokens(token_ids, space_token_id=3) is True

    def test_consecutive_spaces_in_middle(self):
        """Consecutive space tokens in the middle should return True."""
        token_ids = (1, 2, 3, 3, 4)  # 3 is space token
        assert has_consecutive_spaces_tokens(token_ids, space_token_id=3) is True

    def test_consecutive_spaces_at_end(self):
        """Consecutive space tokens at the end should return True."""
        token_ids = (1, 2, 4, 3, 3)  # 3 is space token
        assert has_consecutive_spaces_tokens(token_ids, space_token_id=3) is True

    def test_non_consecutive_spaces(self):
        """Non-consecutive space tokens should return False."""
        token_ids = (1, 3, 2, 3, 4)
        assert has_consecutive_spaces_tokens(token_ids, space_token_id=3) is False

    def test_single_space(self):
        """Single space token should return False."""
        token_ids = (1, 2, 3, 4, 5)
        assert has_consecutive_spaces_tokens(token_ids, space_token_id=3) is False

    def test_no_spaces(self):
        """No space tokens should return False."""
        token_ids = (1, 2, 4, 5, 6)
        assert has_consecutive_spaces_tokens(token_ids, space_token_id=3) is False

    def test_empty_sequence(self):
        """Empty sequence should return False."""
        token_ids = ()
        assert has_consecutive_spaces_tokens(token_ids, space_token_id=3) is False

    def test_single_token(self):
        """Single token should return False."""
        token_ids = (3,)
        assert has_consecutive_spaces_tokens(token_ids, space_token_id=3) is False

    def test_triple_consecutive_spaces(self):
        """Three consecutive space tokens should return True."""
        token_ids = (1, 3, 3, 3, 4)
        assert has_consecutive_spaces_tokens(token_ids, space_token_id=3) is True


class TestDoubleSpacePattern:
    """Test the pre-compiled DOUBLE_SPACE_PATTERN regex."""

    def test_double_space_matches(self):
        """Double space in string should be found."""
        assert DOUBLE_SPACE_PATTERN.findall("hello  world") != []

    def test_single_space_no_match(self):
        """Single space should not match."""
        assert DOUBLE_SPACE_PATTERN.findall("hello world") == []

    def test_no_space_no_match(self):
        """No space should not match."""
        assert DOUBLE_SPACE_PATTERN.findall("helloworld") == []

    def test_triple_space_matches(self):
        """Triple space should match (contains double space)."""
        assert DOUBLE_SPACE_PATTERN.findall("hello   world") != []

    def test_leading_double_space(self):
        """Leading double space should match."""
        assert DOUBLE_SPACE_PATTERN.findall("  hello") != []

    def test_trailing_double_space(self):
        """Trailing double space should match."""
        assert DOUBLE_SPACE_PATTERN.findall("hello  ") != []


class TestBeamsearchMethods:
    """Test Beamsearch methods that don't require GPU."""

    @pytest.fixture
    def mock_beamsearch(self):
        """Create a Beamsearch instance with minimal initialization."""
        # Create instance without calling __init__ to avoid model loading
        bs = object.__new__(Beamsearch)
        bs.gpu_batch_size = 100
        bs.use_token_mode = False
        bs.use_kv_cache = False
        bs.space_token_id = None
        return bs

    def test_next_step_generates_candidates(self, mock_beamsearch):
        """next_step should generate all possible space insertions."""
        candidates = mock_beamsearch.next_step(["abc"])
        # Should include: "abc", "a bc", "ab c"
        assert "abc" in candidates
        assert "a bc" in candidates
        assert "ab c" in candidates

    def test_next_step_filters_double_spaces(self, mock_beamsearch):
        """next_step should not generate double-space candidates."""
        candidates = mock_beamsearch.next_step(["a bc"])
        # Should not contain "a  bc" (double space)
        assert all("  " not in c for c in candidates)

    def test_next_step_empty_list(self, mock_beamsearch):
        """next_step with empty list should return empty list."""
        candidates = mock_beamsearch.next_step([])
        assert candidates == []

    def test_next_step_single_char(self, mock_beamsearch):
        """next_step with single character should return that character."""
        candidates = mock_beamsearch.next_step(["a"])
        assert candidates == ["a"]

    def test_reshape_tree_basic(self, mock_beamsearch):
        """reshape_tree should chunk list into sublists."""
        tree = ["a", "b", "c", "d", "e"]
        reshaped = mock_beamsearch.reshape_tree(tree, 2)
        assert reshaped == [["a", "b"], ["c", "d"], ["e"]]

    def test_reshape_tree_exact_division(self, mock_beamsearch):
        """reshape_tree with exact division should have equal chunks."""
        tree = ["a", "b", "c", "d"]
        reshaped = mock_beamsearch.reshape_tree(tree, 2)
        assert reshaped == [["a", "b"], ["c", "d"]]

    def test_reshape_tree_larger_measure(self, mock_beamsearch):
        """reshape_tree with measure larger than list should return single chunk."""
        tree = ["a", "b", "c"]
        reshaped = mock_beamsearch.reshape_tree(tree, 10)
        assert reshaped == [["a", "b", "c"]]

    def test_reshape_tree_measure_one(self, mock_beamsearch):
        """reshape_tree with measure=1 should create single-element chunks."""
        tree = ["a", "b", "c"]
        reshaped = mock_beamsearch.reshape_tree(tree, 1)
        assert reshaped == [["a"], ["b"], ["c"]]

    def test_reshape_tree_empty(self, mock_beamsearch):
        """reshape_tree with empty list should return empty list."""
        tree = []
        reshaped = mock_beamsearch.reshape_tree(tree, 2)
        assert reshaped == []

    def test_flatten_list_basic(self, mock_beamsearch):
        """flatten_list should flatten nested lists."""
        nested = [["a", "b"], ["c", "d"]]
        flat = mock_beamsearch.flatten_list(nested)
        assert flat == ["a", "b", "c", "d"]

    def test_flatten_list_uneven(self, mock_beamsearch):
        """flatten_list should handle uneven sublists."""
        nested = [["a", "b"], ["c"], ["d", "e", "f"]]
        flat = mock_beamsearch.flatten_list(nested)
        assert flat == ["a", "b", "c", "d", "e", "f"]

    def test_flatten_list_empty_sublists(self, mock_beamsearch):
        """flatten_list should handle empty sublists."""
        nested = [["a"], [], ["b"]]
        flat = mock_beamsearch.flatten_list(nested)
        assert flat == ["a", "b"]

    def test_flatten_list_empty(self, mock_beamsearch):
        """flatten_list with empty list should return empty list."""
        nested = []
        flat = mock_beamsearch.flatten_list(nested)
        assert flat == []

    def test_trim_tree_keeps_topk(self, mock_beamsearch):
        """trim_tree should keep only top-k candidates per character sequence."""
        tree = ["helloworld", "hello world", "h elloworld", "he lloworld"]
        prob_dict = {
            "helloworld": 0.5,
            "hello world": 0.1,  # best score (lowest)
            "h elloworld": 0.3,
            "he lloworld": 0.2
        }
        trimmed = mock_beamsearch.trim_tree(tree, prob_dict, topk=1)
        # Should only keep the best score per character sequence
        assert "hello world" in trimmed
        assert len(trimmed) == 1

    def test_trim_tree_topk_two(self, mock_beamsearch):
        """trim_tree with topk=2 should keep two best candidates."""
        tree = ["helloworld", "hello world", "h elloworld", "he lloworld"]
        prob_dict = {
            "helloworld": 0.5,
            "hello world": 0.1,
            "h elloworld": 0.3,
            "he lloworld": 0.2
        }
        trimmed = mock_beamsearch.trim_tree(tree, prob_dict, topk=2)
        assert "hello world" in trimmed
        assert "he lloworld" in trimmed
        assert len(trimmed) == 2

    def test_trim_tree_multiple_char_sequences(self, mock_beamsearch):
        """trim_tree should handle multiple character sequences."""
        tree = ["abc", "a bc", "xyz", "x yz"]
        prob_dict = {
            "abc": 0.3,
            "a bc": 0.1,
            "xyz": 0.4,
            "x yz": 0.2
        }
        trimmed = mock_beamsearch.trim_tree(tree, prob_dict, topk=1)
        # Should keep best for each: "a bc" for "abc", "x yz" for "xyz"
        assert "a bc" in trimmed
        assert "x yz" in trimmed
        assert len(trimmed) == 2


class TestTrimNodes:
    """Test the trim_nodes method for token-based operations."""

    @pytest.fixture
    def mock_beamsearch(self):
        """Create a Beamsearch instance for testing."""
        bs = object.__new__(Beamsearch)
        bs.gpu_batch_size = 100
        return bs

    def test_trim_nodes_basic(self, mock_beamsearch):
        """trim_nodes should keep top-k nodes per character sequence."""
        nodes = [
            Node(hypothesis="hello world", characters="helloworld", score=0.1),
            Node(hypothesis="helloworld", characters="helloworld", score=0.5),
            Node(hypothesis="h elloworld", characters="helloworld", score=0.3),
        ]
        trimmed = mock_beamsearch.trim_nodes(nodes, topk=1)
        assert len(trimmed) == 1
        assert trimmed[0].hypothesis == "hello world"

    def test_trim_nodes_multiple_sequences(self, mock_beamsearch):
        """trim_nodes should work with multiple character sequences."""
        nodes = [
            Node(hypothesis="a bc", characters="abc", score=0.1),
            Node(hypothesis="abc", characters="abc", score=0.5),
            Node(hypothesis="x yz", characters="xyz", score=0.2),
            Node(hypothesis="xyz", characters="xyz", score=0.4),
        ]
        trimmed = mock_beamsearch.trim_nodes(nodes, topk=1)
        assert len(trimmed) == 2
        chars = {n.characters for n in trimmed}
        assert chars == {"abc", "xyz"}

    def test_trim_nodes_preserves_fields(self, mock_beamsearch):
        """trim_nodes should preserve all Node fields."""
        nodes = [
            Node(
                hypothesis="test",
                characters="test",
                score=0.1,
                token_ids=(1, 2, 3),
                past_key_values=None
            ),
        ]
        trimmed = mock_beamsearch.trim_nodes(nodes, topk=1)
        assert trimmed[0].token_ids == (1, 2, 3)


class TestModelSupportsKVCache:
    """Test the _model_supports_kv_cache method."""

    @pytest.fixture
    def mock_beamsearch(self):
        """Create a Beamsearch instance for testing."""
        bs = object.__new__(Beamsearch)
        return bs

    def test_gpt2_supports_kv_cache(self, mock_beamsearch):
        """GPT2 should support KV-caching."""
        assert mock_beamsearch._model_supports_kv_cache("gpt2") is True

    def test_incremental_supports_kv_cache(self, mock_beamsearch):
        """Incremental models should support KV-caching."""
        assert mock_beamsearch._model_supports_kv_cache("incremental") is True

    def test_bert_no_kv_cache(self, mock_beamsearch):
        """BERT should not support KV-caching."""
        assert mock_beamsearch._model_supports_kv_cache("bert") is False

    def test_masked_no_kv_cache(self, mock_beamsearch):
        """Masked LM should not support KV-caching."""
        assert mock_beamsearch._model_supports_kv_cache("masked") is False

    def test_seq2seq_no_kv_cache(self, mock_beamsearch):
        """Seq2Seq should not support KV-caching."""
        assert mock_beamsearch._model_supports_kv_cache("seq2seq") is False

    def test_case_insensitive(self, mock_beamsearch):
        """Model type check should be case insensitive."""
        assert mock_beamsearch._model_supports_kv_cache("GPT2") is True
        assert mock_beamsearch._model_supports_kv_cache("Incremental") is True

