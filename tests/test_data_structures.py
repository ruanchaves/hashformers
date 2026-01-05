"""Tests for beamsearch data structures module.

HASH-409: Improve Test Coverage - Phase 1.2
Tests for hashformers.beamsearch.data_structures
"""

import pytest
import pandas as pd
from hashformers.beamsearch.data_structures import (
    Node,
    ProbabilityDictionary,
    enforce_prob_dict
)


class TestNode:
    """Test the Node dataclass."""

    def test_node_creation_basic(self):
        """Node should be created with required fields."""
        node = Node(hypothesis="hello world", characters="helloworld", score=0.5)
        assert node.hypothesis == "hello world"
        assert node.characters == "helloworld"
        assert node.score == 0.5

    def test_node_optional_fields_default_none(self):
        """Optional fields should default to None."""
        node = Node(hypothesis="test", characters="test", score=0.0)
        assert node.token_ids is None
        assert node.past_key_values is None

    def test_node_with_token_ids(self):
        """Node should accept token_ids."""
        node = Node(
            hypothesis="hello",
            characters="hello",
            score=0.5,
            token_ids=(1, 2, 3, 4)
        )
        assert node.token_ids == (1, 2, 3, 4)

    def test_node_with_past_key_values(self):
        """Node should accept past_key_values."""
        mock_cache = (("layer1",), ("layer2",))
        node = Node(
            hypothesis="hello",
            characters="hello",
            score=0.5,
            past_key_values=mock_cache
        )
        assert node.past_key_values == mock_cache

    def test_node_score_can_be_negative(self):
        """Node score can be negative (log probabilities)."""
        node = Node(hypothesis="test", characters="test", score=-2.5)
        assert node.score == -2.5

    def test_node_score_can_be_zero(self):
        """Node score can be zero."""
        node = Node(hypothesis="test", characters="test", score=0.0)
        assert node.score == 0.0


class TestProbabilityDictionary:
    """Test the ProbabilityDictionary dataclass."""

    @pytest.fixture
    def sample_prob_dict(self):
        """Create a sample ProbabilityDictionary for testing."""
        return ProbabilityDictionary({
            "hello world": 0.1,
            "helloworld": 0.5,
            "he lloworld": 0.3,
            "h elloworld": 0.4,
            "test case": 0.2,
            "testcase": 0.6
        })

    def test_to_dataframe_structure(self, sample_prob_dict):
        """to_dataframe should return DataFrame with correct columns."""
        df = sample_prob_dict.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "characters" in df.columns
        assert "segmentation" in df.columns
        assert "score" in df.columns

    def test_to_dataframe_content(self, sample_prob_dict):
        """to_dataframe should contain all entries."""
        df = sample_prob_dict.to_dataframe()
        assert len(df) == 6
        assert "hello world" in df["segmentation"].values

    def test_to_dataframe_custom_field_names(self, sample_prob_dict):
        """to_dataframe should accept custom field names."""
        df = sample_prob_dict.to_dataframe(
            characters_field="chars",
            segmentation_field="seg",
            score_field="prob"
        )
        assert "chars" in df.columns
        assert "seg" in df.columns
        assert "prob" in df.columns

    def test_to_dataframe_sorted(self, sample_prob_dict):
        """to_dataframe should be sorted by characters and score."""
        df = sample_prob_dict.to_dataframe()
        # Check that it's sorted (characters first, then score)
        assert df.iloc[0]["characters"] <= df.iloc[-1]["characters"]

    def test_get_top_k_returns_dict(self, sample_prob_dict):
        """get_top_k should return dict by default."""
        result = sample_prob_dict.get_top_k(k=1)
        assert isinstance(result, dict)

    def test_get_top_k_returns_dataframe(self, sample_prob_dict):
        """get_top_k should return DataFrame when requested."""
        result = sample_prob_dict.get_top_k(k=1, return_dataframe=True)
        assert isinstance(result, pd.DataFrame)

    def test_get_top_k_keeps_lowest_scores(self, sample_prob_dict):
        """get_top_k should keep lowest scores (best in LM scoring)."""
        result = sample_prob_dict.get_top_k(k=1)
        # For "helloworld" character sequence, "hello world" has score 0.1 (lowest)
        assert "hello world" in result

    def test_get_top_k_multiple_per_group(self, sample_prob_dict):
        """get_top_k with k=2 should return multiple per group."""
        result = sample_prob_dict.get_top_k(k=2, return_dataframe=True)
        # Should have entries for each character group
        helloworld_entries = result[result["characters"] == "helloworld"]
        assert len(helloworld_entries) == 2

    def test_get_top_k_with_fill(self, sample_prob_dict):
        """get_top_k with fill=True should pad groups to k entries."""
        result = sample_prob_dict.get_top_k(k=3, return_dataframe=True, fill=True)
        # Each group should have exactly 3 entries
        group_sizes = result.groupby("characters").size()
        assert all(group_sizes == 3)

    def test_get_segmentations_dict(self, sample_prob_dict):
        """get_segmentations should return dict by default."""
        segs = sample_prob_dict.get_segmentations(astype='dict')
        assert isinstance(segs, dict)
        # Keys should be character sequences
        assert "helloworld" in segs

    def test_get_segmentations_list(self, sample_prob_dict):
        """get_segmentations should return list when requested."""
        segs = sample_prob_dict.get_segmentations(astype='list')
        assert isinstance(segs, list)

    def test_get_segmentations_with_gold_array(self, sample_prob_dict):
        """get_segmentations should align with gold_array when provided."""
        gold = ["hello world", "test case"]
        segs = sample_prob_dict.get_segmentations(astype='list', gold_array=gold)
        assert isinstance(segs, list)
        assert len(segs) == 2


class TestEnforceProbDict:
    """Test the enforce_prob_dict function."""

    def test_passthrough_prob_dict(self):
        """ProbabilityDictionary should pass through unchanged."""
        pd_obj = ProbabilityDictionary({"test": 0.5})
        result = enforce_prob_dict(pd_obj)
        assert result is pd_obj

    def test_from_dict(self):
        """Regular dict should be converted to ProbabilityDictionary."""
        result = enforce_prob_dict({"test": 0.5, "hello": 0.3})
        assert isinstance(result, ProbabilityDictionary)
        assert result.dictionary["test"] == 0.5

    def test_from_list_of_strings(self):
        """List of strings should be converted with zero scores."""
        result = enforce_prob_dict(["hello", "world", "test"])
        assert isinstance(result, ProbabilityDictionary)
        assert result.dictionary["hello"] == 0.0
        assert result.dictionary["world"] == 0.0

    def test_from_list_deduplicates(self):
        """List conversion should deduplicate entries."""
        result = enforce_prob_dict(["hello", "hello", "world"])
        assert len(result.dictionary) == 2

    def test_from_dataframe(self):
        """DataFrame should be converted to ProbabilityDictionary."""
        df = pd.DataFrame({
            "segmentation": ["hello world", "test case"],
            "score": [0.5, 0.3]
        })
        result = enforce_prob_dict(df)
        assert isinstance(result, ProbabilityDictionary)
        assert result.dictionary["hello world"] == 0.5

    def test_from_dataframe_custom_fields(self):
        """DataFrame conversion should accept custom field names."""
        df = pd.DataFrame({
            "seg": ["hello world"],
            "prob": [0.5]
        })
        result = enforce_prob_dict(df, score_field="prob", segmentation_field="seg")
        assert result.dictionary["hello world"] == 0.5

    def test_unsupported_type_raises(self):
        """Unsupported types should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            enforce_prob_dict(12345)

    def test_from_empty_dict(self):
        """Empty dict should be converted successfully."""
        result = enforce_prob_dict({})
        assert isinstance(result, ProbabilityDictionary)
        assert len(result.dictionary) == 0

    def test_from_empty_list(self):
        """Empty list should be converted successfully."""
        result = enforce_prob_dict([])
        assert isinstance(result, ProbabilityDictionary)
        assert len(result.dictionary) == 0


class TestProbabilityDictionaryIO:
    """Test ProbabilityDictionary I/O methods."""

    @pytest.fixture
    def sample_prob_dict(self):
        """Create a sample ProbabilityDictionary."""
        return ProbabilityDictionary({
            "hello world": 0.1,
            "test case": 0.2
        })

    def test_to_csv(self, sample_prob_dict, tmp_path):
        """to_csv should write CSV file."""
        filepath = tmp_path / "test.csv"
        sample_prob_dict.to_csv(str(filepath))
        assert filepath.exists()
        # Read back and verify
        df = pd.read_csv(filepath)
        assert "segmentation" in df.columns

    def test_to_json(self, sample_prob_dict, tmp_path):
        """to_json should write JSON file."""
        import json
        filepath = tmp_path / "test.json"
        sample_prob_dict.to_json(str(filepath))
        assert filepath.exists()
        # Read back and verify
        with open(filepath) as f:
            data = json.load(f)
        assert data["hello world"] == 0.1

