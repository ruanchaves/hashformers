"""Tests for ensemble module.

HASH-409: Improve Test Coverage - Phase 1.5
Tests for hashformers.ensemble.top2_fusion

HASH-410: Added tests for TopKEnsembler and generalized top-k fusion
"""

import pytest
import numpy as np
import pandas as pd
from hashformers.ensemble.top2_fusion import (
    run_ensemble, 
    Top2Ensembler, 
    top2_ensemble,
    TopKEnsembler,
    topk_ensemble,
    run_weighted_ensemble,
)
from hashformers.experiments.utils import (
    filter_and_project_scores,
    calculate_diff_scores,
    calculate_weighted_scores,
    build_ensemble_df,
    build_ensemble_df_topk,
)


class TestRunEnsemble:
    """Test the run_ensemble function."""

    def test_basic_ensemble(self):
        """run_ensemble should compute ensemble output."""
        a_diff = np.array([0.1, 0.2])
        b_diff = np.array([0.3, 0.1])
        a_rank = np.array([0, 1])
        b_rank = np.array([1, 0])

        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=0.2, beta=0.1)
        assert len(result) == 2
        assert isinstance(result, np.ndarray)

    def test_zero_weights_uses_a_rank(self):
        """With zero weights, delta=0, so should use a_rank."""
        a_diff = np.array([0.1, 0.2])
        b_diff = np.array([0.3, 0.1])
        a_rank = np.array([0, 1])
        b_rank = np.array([1, 0])

        # With zero weights, delta = 0, decision = 0, so output = a_rank
        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=0.0, beta=0.0)
        np.testing.assert_array_equal(result, a_rank)

    def test_high_alpha_low_beta(self):
        """High alpha, low beta should favor a's scoring."""
        a_diff = np.array([0.5, 0.5])
        b_diff = np.array([0.1, 0.1])
        a_rank = np.array([0, 1])
        b_rank = np.array([1, 0])

        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=1.0, beta=0.0)
        # delta = alpha * a_diff - beta * b_diff = 0.5 - 0 = 0.5 > 0
        # decision = 0, so output = a_rank
        np.testing.assert_array_equal(result, a_rank)

    def test_high_beta_low_alpha(self):
        """High beta, low alpha should favor b's scoring when delta < 0."""
        a_diff = np.array([0.1, 0.1])
        b_diff = np.array([0.5, 0.5])
        a_rank = np.array([0, 1])
        b_rank = np.array([1, 0])

        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=0.0, beta=1.0)
        # delta = 0 - 0.5 = -0.5 < 0
        # decision = 1, so output = b_rank
        np.testing.assert_array_equal(result, b_rank)

    def test_mixed_decisions(self):
        """Ensemble should make different decisions per position."""
        a_diff = np.array([0.5, 0.1])
        b_diff = np.array([0.1, 0.5])
        a_rank = np.array([0, 0])
        b_rank = np.array([1, 1])

        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=0.2, beta=0.2)
        # delta[0] = 0.2*0.5 - 0.2*0.1 = 0.1 - 0.02 = 0.08 > 0, use a_rank[0] = 0
        # delta[1] = 0.2*0.1 - 0.2*0.5 = 0.02 - 0.1 = -0.08 < 0, use b_rank[1] = 1
        assert result[0] == 0
        assert result[1] == 1

    def test_single_element(self):
        """run_ensemble should work with single element arrays."""
        a_diff = np.array([0.5])
        b_diff = np.array([0.3])
        a_rank = np.array([0])
        b_rank = np.array([1])

        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=0.2, beta=0.1)
        assert len(result) == 1

    def test_large_arrays(self):
        """run_ensemble should handle large arrays efficiently."""
        n = 10000
        a_diff = np.random.rand(n)
        b_diff = np.random.rand(n)
        a_rank = np.random.randint(0, 2, n)
        b_rank = np.random.randint(0, 2, n)

        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=0.2, beta=0.1)
        assert len(result) == n

    def test_negative_diffs(self):
        """run_ensemble should handle negative difference values."""
        a_diff = np.array([-0.1, -0.2])
        b_diff = np.array([-0.3, -0.1])
        a_rank = np.array([0, 1])
        b_rank = np.array([1, 0])

        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=0.2, beta=0.1)
        assert len(result) == 2


class TestTop2Ensembler:
    """Test the Top2Ensembler class."""

    def test_init(self):
        """Top2Ensembler should initialize successfully."""
        ensembler = Top2Ensembler()
        assert ensembler is not None

    def test_backwards_compatibility_alias(self):
        """Top2_Ensembler alias should work (HASH-012)."""
        from hashformers.ensemble.top2_fusion import Top2_Ensembler
        ensembler = Top2_Ensembler()
        assert ensembler is not None
        assert isinstance(ensembler, Top2Ensembler)

    def test_has_run_method(self):
        """Top2Ensembler should have run method."""
        ensembler = Top2Ensembler()
        assert hasattr(ensembler, 'run')
        assert callable(ensembler.run)

    def test_run_method_signature(self):
        """run method should accept expected parameters."""
        import inspect
        ensembler = Top2Ensembler()
        sig = inspect.signature(ensembler.run)
        params = list(sig.parameters.keys())
        assert 'segmenter_run' in params
        assert 'reranker_run' in params
        assert 'alpha' in params
        assert 'beta' in params

    def test_inherits_from_topk_ensembler(self):
        """Top2Ensembler should inherit from TopKEnsembler (HASH-410)."""
        ensembler = Top2Ensembler()
        assert isinstance(ensembler, TopKEnsembler)
        assert ensembler.k == 2


class TestTopKEnsembler:
    """Test the TopKEnsembler class (HASH-410)."""

    def test_init_default_k(self):
        """TopKEnsembler should default to k=2."""
        ensembler = TopKEnsembler()
        assert ensembler.k == 2

    def test_init_custom_k(self):
        """TopKEnsembler should accept custom k values."""
        ensembler = TopKEnsembler(k=5)
        assert ensembler.k == 5

    def test_init_k_must_be_at_least_2(self):
        """TopKEnsembler should raise error if k < 2."""
        with pytest.raises(ValueError) as excinfo:
            TopKEnsembler(k=1)
        assert "k must be at least 2" in str(excinfo.value)

    def test_has_run_method(self):
        """TopKEnsembler should have run method."""
        ensembler = TopKEnsembler(k=3)
        assert hasattr(ensembler, 'run')
        assert callable(ensembler.run)

    def test_run_method_signature(self):
        """run method should accept expected parameters."""
        import inspect
        ensembler = TopKEnsembler(k=5)
        sig = inspect.signature(ensembler.run)
        params = list(sig.parameters.keys())
        assert 'segmenter_run' in params
        assert 'reranker_run' in params
        assert 'alpha' in params
        assert 'beta' in params

    def test_k2_backward_compatibility(self):
        """TopKEnsembler with k=2 should be compatible with Top2Ensembler."""
        topk = TopKEnsembler(k=2)
        top2 = Top2Ensembler()
        # Both should use pairwise difference logic internally
        assert topk.k == top2.k


class TestTopKEnsembleFunctions:
    """Test the topk_ensemble and related functions (HASH-410)."""

    @pytest.fixture
    def sample_dicts(self):
        """Create sample probability dictionaries for testing."""
        dict_1 = {
            "hello world": 0.1,
            "helloworld": 0.5,
        }
        dict_2 = {
            "hello world": 0.2,
            "helloworld": 0.4,
        }
        return dict_1, dict_2

    def test_topk_ensemble_k2_returns_dataframe(self, sample_dicts):
        """topk_ensemble with k=2 should return a DataFrame."""
        dict_1, dict_2 = sample_dicts
        result = topk_ensemble(dict_1, dict_2, k=2, alpha=0.2, beta=0.1)
        assert isinstance(result, pd.DataFrame)

    def test_topk_ensemble_k2_has_required_columns(self, sample_dicts):
        """topk_ensemble with k=2 should have pairwise columns."""
        dict_1, dict_2 = sample_dicts
        result = topk_ensemble(dict_1, dict_2, k=2, alpha=0.2, beta=0.1)
        assert "diff" in result.columns
        assert "rank" in result.columns


class TestFilterAndProjectScoresWithK:
    """Test filter_and_project_scores with k parameter (HASH-410)."""

    @pytest.fixture
    def sample_dataframes(self):
        """Create sample dataframes for testing."""
        a = pd.DataFrame({
            'hashtag': ['abc', 'abc', 'abc', 'abc', 'xyz', 'xyz', 'xyz', 'xyz'],
            'segmentation': ['a bc', 'ab c', 'abc', 'a b c', 'x yz', 'xy z', 'xyz', 'x y z'],
            'score': [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]
        })
        b = pd.DataFrame({
            'hashtag': ['abc', 'abc', 'abc', 'abc', 'xyz', 'xyz', 'xyz', 'xyz'],
            'segmentation': ['a bc', 'ab c', 'abc', 'a b c', 'x yz', 'xy z', 'xyz', 'x y z'],
            'score': [0.15, 0.25, 0.35, 0.45, 0.15, 0.25, 0.35, 0.45]
        })
        return a, b

    def test_default_k_is_2(self, sample_dataframes):
        """filter_and_project_scores should default to k=2."""
        a, b = sample_dataframes
        result = filter_and_project_scores(a, b)
        # Should have 2 candidates per group (2 groups = 4 total)
        assert len(result[0]) == 4

    def test_k_3(self, sample_dataframes):
        """filter_and_project_scores with k=3 should return 3 per group."""
        a, b = sample_dataframes
        result = filter_and_project_scores(a, b, k=3)
        # Should have 3 candidates per group (2 groups = 6 total)
        assert len(result[0]) == 6

    def test_k_4(self, sample_dataframes):
        """filter_and_project_scores with k=4 should return 4 per group."""
        a, b = sample_dataframes
        result = filter_and_project_scores(a, b, k=4)
        # Should have 4 candidates per group (2 groups = 8 total)
        assert len(result[0]) == 8


class TestCalculateDiffScores:
    """Test calculate_diff_scores with k parameter (HASH-410)."""

    @pytest.fixture
    def sample_k2_dataframes(self):
        """Create sample dataframes with exactly 2 candidates per group."""
        a = pd.DataFrame({
            'hashtag': ['abc', 'abc', 'xyz', 'xyz'],
            'segmentation': ['a bc', 'ab c', 'x yz', 'xy z'],
            'score': [0.1, 0.2, 0.1, 0.2]
        })
        b = pd.DataFrame({
            'hashtag': ['abc', 'abc', 'xyz', 'xyz'],
            'segmentation': ['a bc', 'ab c', 'x yz', 'xy z'],
            'score': [0.15, 0.25, 0.15, 0.25]
        })
        return a, b

    def test_k2_works(self, sample_k2_dataframes):
        """calculate_diff_scores should work with k=2."""
        a, b = sample_k2_dataframes
        result = calculate_diff_scores(a, b, k=2)
        assert len(result) == 2
        assert 'diff' in result[0].columns
        assert 'rank' in result[0].columns

    def test_k_greater_than_2_raises_error(self, sample_k2_dataframes):
        """calculate_diff_scores should raise error for k>2."""
        a, b = sample_k2_dataframes
        with pytest.raises(ValueError) as excinfo:
            calculate_diff_scores(a, b, k=3)
        assert "calculate_diff_scores only supports k=2" in str(excinfo.value)


class TestCalculateWeightedScores:
    """Test calculate_weighted_scores function (HASH-410)."""

    @pytest.fixture
    def sample_dataframes(self):
        """Create sample dataframes for testing."""
        a = pd.DataFrame({
            'hashtag': ['abc', 'abc', 'abc', 'xyz', 'xyz', 'xyz'],
            'segmentation': ['a bc', 'ab c', 'abc', 'x yz', 'xy z', 'xyz'],
            'score': [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
        })
        b = pd.DataFrame({
            'hashtag': ['abc', 'abc', 'abc', 'xyz', 'xyz', 'xyz'],
            'segmentation': ['a bc', 'ab c', 'abc', 'x yz', 'xy z', 'xyz'],
            'score': [0.3, 0.1, 0.2, 0.3, 0.1, 0.2]
        })
        return a, b

    def test_returns_dataframe(self, sample_dataframes):
        """calculate_weighted_scores should return a DataFrame."""
        a, b = sample_dataframes
        result = calculate_weighted_scores(a, b, k=3)
        assert isinstance(result, pd.DataFrame)

    def test_has_weighted_columns(self, sample_dataframes):
        """calculate_weighted_scores should add weighted columns."""
        a, b = sample_dataframes
        result = calculate_weighted_scores(a, b, k=3)
        assert 'score_a' in result.columns
        assert 'score_b' in result.columns
        assert 'weighted_score' in result.columns
        assert 'weighted_rank' in result.columns

    def test_weighted_score_calculation(self, sample_dataframes):
        """weighted_score should be alpha*a + beta*b."""
        a, b = sample_dataframes
        result = calculate_weighted_scores(a, b, k=3, alpha=0.6, beta=0.4)
        # Check first row: 0.6 * 0.1 + 0.4 * 0.3 = 0.06 + 0.12 = 0.18
        first_row = result.iloc[0]
        expected_score = 0.6 * first_row['score_a'] + 0.4 * first_row['score_b']
        assert abs(first_row['weighted_score'] - expected_score) < 1e-9

    def test_weighted_rank_within_group(self, sample_dataframes):
        """weighted_rank should be 0-based within each group."""
        a, b = sample_dataframes
        result = calculate_weighted_scores(a, b, k=3, alpha=0.5, beta=0.5)
        # Ranks should be 0, 1, 2 within each group
        for _, group in result.groupby('hashtag'):
            ranks = sorted(group['weighted_rank'].tolist())
            assert ranks == [0, 1, 2]


class TestBuildEnsembleDfTopK:
    """Test build_ensemble_df_topk function (HASH-410)."""

    @pytest.fixture
    def sample_dataframes(self):
        """Create sample dataframes for testing."""
        a = pd.DataFrame({
            'hashtag': ['abc', 'abc', 'abc', 'xyz', 'xyz', 'xyz'],
            'segmentation': ['a bc', 'ab c', 'abc', 'x yz', 'xy z', 'xyz'],
            'score': [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
        })
        b = pd.DataFrame({
            'hashtag': ['abc', 'abc', 'abc', 'xyz', 'xyz', 'xyz'],
            'segmentation': ['a bc', 'ab c', 'abc', 'x yz', 'xy z', 'xyz'],
            'score': [0.3, 0.1, 0.2, 0.3, 0.1, 0.2]
        })
        return a, b

    def test_returns_dataframe(self, sample_dataframes):
        """build_ensemble_df_topk should return a DataFrame."""
        a, b = sample_dataframes
        result = build_ensemble_df_topk(a, b, k=3)
        assert isinstance(result, pd.DataFrame)

    def test_has_weighted_rank(self, sample_dataframes):
        """build_ensemble_df_topk should have weighted_rank column."""
        a, b = sample_dataframes
        result = build_ensemble_df_topk(a, b, k=3)
        assert 'weighted_rank' in result.columns

    def test_custom_alpha_beta(self, sample_dataframes):
        """build_ensemble_df_topk should accept custom alpha and beta."""
        a, b = sample_dataframes
        result = build_ensemble_df_topk(a, b, k=3, alpha=0.7, beta=0.3)
        assert isinstance(result, pd.DataFrame)


class TestTop2EnsembleFunction:
    """Test the top2_ensemble function."""

    @pytest.fixture
    def sample_dicts(self):
        """Create sample probability dictionaries for testing."""
        dict_1 = {
            "hello world": 0.1,
            "helloworld": 0.5,
        }
        dict_2 = {
            "hello world": 0.2,
            "helloworld": 0.4,
        }
        return dict_1, dict_2

    def test_returns_dataframe(self, sample_dicts):
        """top2_ensemble should return a DataFrame."""
        dict_1, dict_2 = sample_dicts
        result = top2_ensemble(dict_1, dict_2, alpha=0.2, beta=0.1)
        assert isinstance(result, pd.DataFrame)

    def test_has_ensemble_rank_column(self, sample_dicts):
        """Result should have ensemble_rank column."""
        dict_1, dict_2 = sample_dicts
        result = top2_ensemble(dict_1, dict_2, alpha=0.2, beta=0.1)
        assert "ensemble_rank" in result.columns

    def test_has_required_columns(self, sample_dicts):
        """Result should have all required columns."""
        dict_1, dict_2 = sample_dicts
        result = top2_ensemble(dict_1, dict_2, alpha=0.2, beta=0.1)
        assert "diff" in result.columns
        assert "diff_2" in result.columns
        assert "rank" in result.columns
        assert "rank_2" in result.columns

    def test_custom_alpha_beta(self, sample_dicts):
        """top2_ensemble should accept custom alpha and beta."""
        dict_1, dict_2 = sample_dicts
        result = top2_ensemble(dict_1, dict_2, alpha=0.5, beta=0.3)
        assert isinstance(result, pd.DataFrame)

    def test_zero_weights(self, sample_dicts):
        """top2_ensemble should work with zero weights."""
        dict_1, dict_2 = sample_dicts
        result = top2_ensemble(dict_1, dict_2, alpha=0.0, beta=0.0)
        assert isinstance(result, pd.DataFrame)


class TestEnsembleEdgeCases:
    """Test edge cases for ensemble functions."""

    def test_run_ensemble_empty_arrays(self):
        """run_ensemble should handle empty arrays."""
        result = run_ensemble(
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            alpha=0.2,
            beta=0.1
        )
        assert len(result) == 0

    def test_run_ensemble_all_zeros(self):
        """run_ensemble should handle all-zero inputs."""
        a_diff = np.zeros(5)
        b_diff = np.zeros(5)
        a_rank = np.zeros(5, dtype=int)
        b_rank = np.ones(5, dtype=int)

        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=0.2, beta=0.1)
        # delta = 0, decision = 0, output = a_rank = 0
        np.testing.assert_array_equal(result, a_rank)

    def test_run_ensemble_float_ranks(self):
        """run_ensemble should work with float rank values."""
        a_diff = np.array([0.1, 0.2])
        b_diff = np.array([0.3, 0.1])
        a_rank = np.array([0.5, 1.5])
        b_rank = np.array([1.5, 0.5])

        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=0.2, beta=0.1)
        assert len(result) == 2


class TestRunWeightedEnsemble:
    """Test the run_weighted_ensemble function (HASH-410)."""

    def test_basic_weighted_ensemble(self):
        """run_weighted_ensemble should select best candidate per group."""
        scores = np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.4])  # 2 groups of 3
        ranks = np.array([1, 0, 2, 0, 2, 1])  # rank 0 is best
        k = 3

        result = run_weighted_ensemble(scores, ranks, k)
        assert len(result) == 2  # 2 groups
        # First group: rank 0 is at index 1 (local), second group: rank 0 is at index 0 (local)
        assert result[0] == 1  # Index of best in first group
        assert result[1] == 0  # Index of best in second group

    def test_single_group(self):
        """run_weighted_ensemble should work with single group."""
        scores = np.array([0.5, 0.3, 0.8])
        ranks = np.array([1, 0, 2])
        k = 3

        result = run_weighted_ensemble(scores, ranks, k)
        assert len(result) == 1
        assert result[0] == 1  # Index of rank 0

