"""Tests for utility modules.

HASH-409: Improve Test Coverage - Phase 1.4
Tests for hashformers.utils.filtering and hashformers.utils.twitter
"""

import pytest
import pandas as pd
import numpy as np
from hashformers.utils.filtering import filter_top_k
from hashformers.utils.twitter import (
    extract_hashtags,
    extract_hashtags_with_prefix,
    HASHTAG_PATTERN
)


class TestFilterTopK:
    """Test the filter_top_k function."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'hashtag': ['abc', 'abc', 'abc', 'xyz', 'xyz'],
            'segmentation': ['a bc', 'ab c', 'abc', 'x yz', 'xyz'],
            'score': [0.1, 0.2, 0.3, 0.1, 0.5]
        })

    def test_basic_filtering(self, sample_df):
        """filter_top_k should filter to k entries per group."""
        result = filter_top_k(sample_df, k=2, gold_field='hashtag')
        # Should have 2 per group = 4 total
        assert len(result) == 4

    def test_keeps_lowest_scores(self, sample_df):
        """filter_top_k should keep entries with lowest scores."""
        result = filter_top_k(sample_df, k=1, gold_field='hashtag')
        # Should keep only the lowest score per group
        abc_row = result[result['hashtag'] == 'abc']
        assert len(abc_row) == 1
        assert abc_row['score'].values[0] == 0.1

    def test_filter_k_one(self, sample_df):
        """filter_top_k with k=1 should return one per group."""
        result = filter_top_k(sample_df, k=1, gold_field='hashtag')
        assert len(result) == 2
        group_sizes = result.groupby('hashtag').size()
        assert all(group_sizes == 1)

    def test_filter_k_larger_than_group(self, sample_df):
        """filter_top_k with k larger than group should return all in group."""
        result = filter_top_k(sample_df, k=10, gold_field='hashtag')
        # Should return all entries since k > group sizes
        assert len(result) == 5

    def test_custom_gold_field(self):
        """filter_top_k should work with custom gold_field."""
        df = pd.DataFrame({
            'group': ['a', 'a', 'b', 'b'],
            'segmentation': ['a1', 'a2', 'b1', 'b2'],
            'score': [0.1, 0.2, 0.3, 0.4]
        })
        result = filter_top_k(df, k=1, gold_field='group')
        assert len(result) == 2

    def test_custom_score_field(self):
        """filter_top_k should work with custom score_field."""
        df = pd.DataFrame({
            'hashtag': ['abc', 'abc'],
            'segmentation': ['a bc', 'abc'],
            'probability': [0.1, 0.2]
        })
        result = filter_top_k(df, k=1, gold_field='hashtag', score_field='probability')
        assert len(result) == 1
        assert result['probability'].values[0] == 0.1

    def test_fill_false(self, sample_df):
        """filter_top_k with fill=False should not pad groups."""
        result = filter_top_k(sample_df, k=3, gold_field='hashtag', fill=False)
        # xyz group only has 2 entries
        xyz_count = len(result[result['hashtag'] == 'xyz'])
        assert xyz_count == 2

    def test_fill_true_pads_groups(self, sample_df):
        """filter_top_k with fill=True should pad groups to k entries."""
        result = filter_top_k(sample_df, k=3, gold_field='hashtag', fill=True)
        group_sizes = result.groupby('hashtag').size()
        assert all(group_sizes == 3)

    def test_empty_dataframe(self):
        """filter_top_k should handle empty DataFrame."""
        df = pd.DataFrame(columns=['hashtag', 'segmentation', 'score'])
        result = filter_top_k(df, k=2, gold_field='hashtag')
        assert len(result) == 0

    def test_single_group(self):
        """filter_top_k should work with single group."""
        df = pd.DataFrame({
            'hashtag': ['abc', 'abc', 'abc'],
            'segmentation': ['a bc', 'ab c', 'abc'],
            'score': [0.1, 0.2, 0.3]
        })
        result = filter_top_k(df, k=2, gold_field='hashtag')
        assert len(result) == 2

    def test_preserves_original_dataframe(self, sample_df):
        """filter_top_k should not modify original DataFrame."""
        original_len = len(sample_df)
        filter_top_k(sample_df, k=1, gold_field='hashtag')
        assert len(sample_df) == original_len


class TestExtractHashtags:
    """Test the extract_hashtags function."""

    def test_basic_extraction(self):
        """extract_hashtags should extract hashtags without prefix."""
        text = "Hello #world #python"
        result = extract_hashtags(text)
        assert result == ['world', 'python']

    def test_no_hashtags(self):
        """extract_hashtags should return empty list when no hashtags."""
        text = "No hashtags here"
        result = extract_hashtags(text)
        assert result == []

    def test_empty_string(self):
        """extract_hashtags should handle empty string."""
        result = extract_hashtags("")
        assert result == []

    def test_camel_case_hashtag(self):
        """extract_hashtags should extract CamelCase hashtags."""
        text = "#CamelCase #PascalCase"
        result = extract_hashtags(text)
        assert 'CamelCase' in result
        assert 'PascalCase' in result

    def test_underscore_hashtag(self):
        """extract_hashtags should extract hashtags with underscores."""
        text = "#with_underscore #another_one"
        result = extract_hashtags(text)
        assert 'with_underscore' in result
        assert 'another_one' in result

    def test_numeric_hashtag(self):
        """extract_hashtags should extract hashtags with numbers."""
        text = "#123number #test123 #1a2b3c"
        result = extract_hashtags(text)
        assert '123number' in result
        assert 'test123' in result
        assert '1a2b3c' in result

    def test_consecutive_hashtags(self):
        """extract_hashtags should extract consecutive hashtags."""
        text = "#one#two#three"
        result = extract_hashtags(text)
        assert 'one' in result
        assert 'two' in result
        assert 'three' in result

    def test_hashtag_at_start(self):
        """extract_hashtags should extract hashtag at start of string."""
        text = "#first is here"
        result = extract_hashtags(text)
        assert result == ['first']

    def test_hashtag_at_end(self):
        """extract_hashtags should extract hashtag at end of string."""
        text = "Check this out #last"
        result = extract_hashtags(text)
        assert result == ['last']

    def test_only_hashtag_symbol(self):
        """extract_hashtags should not match lone # symbol."""
        text = "Just a # symbol"
        result = extract_hashtags(text)
        assert result == []

    def test_unicode_hashtag(self):
        """extract_hashtags should handle unicode characters."""
        text = "#café #日本語"
        result = extract_hashtags(text)
        # \w with UNICODE flag should match these
        assert len(result) >= 1


class TestExtractHashtagsWithPrefix:
    """Test the extract_hashtags_with_prefix function."""

    def test_basic_extraction(self):
        """extract_hashtags_with_prefix should include # prefix."""
        text = "Hello #world #python"
        result = extract_hashtags_with_prefix(text)
        assert result == ['#world', '#python']

    def test_no_hashtags(self):
        """extract_hashtags_with_prefix should return empty list when no hashtags."""
        text = "No hashtags here"
        result = extract_hashtags_with_prefix(text)
        assert result == []

    def test_empty_string(self):
        """extract_hashtags_with_prefix should handle empty string."""
        result = extract_hashtags_with_prefix("")
        assert result == []

    def test_single_hashtag(self):
        """extract_hashtags_with_prefix should work with single hashtag."""
        text = "Just #one"
        result = extract_hashtags_with_prefix(text)
        assert result == ['#one']

    def test_preserves_case(self):
        """extract_hashtags_with_prefix should preserve original case."""
        text = "#CamelCase #lowercase #UPPERCASE"
        result = extract_hashtags_with_prefix(text)
        assert '#CamelCase' in result
        assert '#lowercase' in result
        assert '#UPPERCASE' in result


class TestHashtagPattern:
    """Test the pre-compiled HASHTAG_PATTERN regex."""

    def test_pattern_matches_simple(self):
        """HASHTAG_PATTERN should match simple hashtags."""
        matches = HASHTAG_PATTERN.findall("#hello")
        assert matches == ['hello']

    def test_pattern_captures_without_hash(self):
        """HASHTAG_PATTERN should capture text without # prefix."""
        matches = HASHTAG_PATTERN.findall("#test")
        assert '#' not in matches[0]

    def test_pattern_handles_multiple(self):
        """HASHTAG_PATTERN should find multiple hashtags."""
        matches = HASHTAG_PATTERN.findall("#one #two #three")
        assert len(matches) == 3

