# HASH-409: Improve Test Coverage Across Core Modules

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Type**    | Task                                                      |
| **Priority**| Medium                                                    |
| **Epic**    | Epic 4: Code Quality & Maintainability                    |
| **Files**   | `tests/test_segmenter.py`, new test files                 |

## Description

The current test suite in `tests/test_segmenter.py` provides only minimal coverage, with just 8 tests focused primarily on the `segmenter` module. Large portions of the codebase have **zero test coverage**, including critical modules like `beamsearch`, `ensemble`, `config`, `utils`, `experiments`, and all `integrations`.

### Current State

**Existing tests (8 total):**
- `test_cuda_availability` - CUDA check
- `test_word_segmenter_output` - BaseWordSegmenter (requires GPU)
- `test_twitter_text_matcher` - TwitterTextMatcher
- `test_regex_word_segmentation` - RegexWordSegmenter
- `test_hashtag_container` - TweetSegmenter.build_hashtag_container()
- `test_tweet_segmentation` - TweetSegmenter.segmented_tweet_generator()
- `test_tweet_segmenter_output_format` - TweetSegmenter.predict()

> [!CAUTION]
> Most core logic in `beamsearch/algorithm.py`, `ensemble/top2_fusion.py`, and `utils/` has no test coverage. Regressions can easily slip through undetected.

### Coverage Gaps by Module

| Module | Coverage | Priority |
|--------|----------|----------|
| `beamsearch/algorithm.py` | ❌ None | High |
| `beamsearch/data_structures.py` | ❌ None | High |
| `beamsearch/reranker.py` | ❌ None | Medium |
| `config.py` | ❌ None | Medium |
| `ensemble/top2_fusion.py` | ❌ None | High |
| `utils/filtering.py` | ❌ None | Medium |
| `utils/twitter.py` | ❌ None | Low |
| `experiments/utils.py` | ❌ None | Low |
| `integrations/*` | ❌ None | Low |

## Tasks

### Phase 1: Unit Tests for Pure Logic (No GPU Required)

These tests can run on any CI/CD system without GPU requirements.

#### 1.1 `tests/test_beamsearch.py` - Beam Search Algorithm

```python
import pytest
from hashformers.beamsearch.algorithm import (
    Beamsearch, 
    has_consecutive_spaces_tokens,
    DOUBLE_SPACE_PATTERN
)
from hashformers.beamsearch.data_structures import Node, ProbabilityDictionary

class TestBeamsearchHelpers:
    """Test helper functions that don't require model loading."""
    
    def test_has_consecutive_spaces_tokens_true(self):
        token_ids = (1, 2, 3, 3, 4)  # 3 is space token
        assert has_consecutive_spaces_tokens(token_ids, space_token_id=3) is True
    
    def test_has_consecutive_spaces_tokens_false(self):
        token_ids = (1, 3, 2, 3, 4)
        assert has_consecutive_spaces_tokens(token_ids, space_token_id=3) is False
    
    def test_double_space_pattern_matches(self):
        assert DOUBLE_SPACE_PATTERN.findall("hello  world") != []
    
    def test_double_space_pattern_no_match(self):
        assert DOUBLE_SPACE_PATTERN.findall("hello world") == []


class TestBeamsearchMethods:
    """Test Beamsearch methods using mocked models."""
    
    @pytest.fixture
    def mock_beamsearch(self, mocker):
        # Mock the model loading to avoid GPU requirement
        mocker.patch('hashformers.beamsearch.algorithm.ModelLM.__init__')
        bs = Beamsearch.__new__(Beamsearch)
        bs.gpu_batch_size = 100
        return bs
    
    def test_next_step_generates_candidates(self, mock_beamsearch):
        candidates = mock_beamsearch.next_step(["hello"])
        assert "hello" in candidates
        assert "h ello" in candidates
        assert "he llo" in candidates
    
    def test_next_step_filters_double_spaces(self, mock_beamsearch):
        candidates = mock_beamsearch.next_step(["h ello"])
        # Should not contain "h  ello" (double space)
        assert all("  " not in c for c in candidates)
    
    def test_reshape_tree(self, mock_beamsearch):
        tree = ["a", "b", "c", "d", "e"]
        reshaped = mock_beamsearch.reshape_tree(tree, 2)
        assert reshaped == [["a", "b"], ["c", "d"], ["e"]]
    
    def test_flatten_list(self, mock_beamsearch):
        nested = [["a", "b"], ["c", "d"]]
        flat = mock_beamsearch.flatten_list(nested)
        assert flat == ["a", "b", "c", "d"]
    
    def test_trim_tree_keeps_topk(self, mock_beamsearch):
        tree = ["helloworld", "hello world", "h elloworld"]
        prob_dict = {
            "helloworld": 0.5,
            "hello world": 0.1,  # best score
            "h elloworld": 0.3
        }
        trimmed = mock_beamsearch.trim_tree(tree, prob_dict, topk=1)
        assert "hello world" in trimmed
```

#### 1.2 `tests/test_data_structures.py` - Data Structures

```python
import pytest
import pandas as pd
from hashformers.beamsearch.data_structures import (
    Node, 
    ProbabilityDictionary, 
    enforce_prob_dict
)

class TestNode:
    def test_node_creation(self):
        node = Node(hypothesis="hello world", characters="helloworld", score=0.5)
        assert node.hypothesis == "hello world"
        assert node.characters == "helloworld"
        assert node.score == 0.5
    
    def test_node_optional_fields(self):
        node = Node(hypothesis="test", characters="test", score=0.0)
        assert node.token_ids is None
        assert node.past_key_values is None


class TestProbabilityDictionary:
    @pytest.fixture
    def sample_prob_dict(self):
        return ProbabilityDictionary({
            "hello world": 0.1,
            "helloworld": 0.5,
            "he lloworld": 0.3,
            "test case": 0.2,
            "testcase": 0.4
        })
    
    def test_to_dataframe(self, sample_prob_dict):
        df = sample_prob_dict.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "characters" in df.columns
        assert "segmentation" in df.columns
        assert "score" in df.columns
    
    def test_get_top_k(self, sample_prob_dict):
        top1 = sample_prob_dict.get_top_k(k=1)
        # Should return best score per character sequence
        assert "hello world" in top1
    
    def test_get_segmentations_dict(self, sample_prob_dict):
        segs = sample_prob_dict.get_segmentations(astype='dict')
        assert isinstance(segs, dict)
        assert "helloworld" in segs
    
    def test_get_segmentations_list(self, sample_prob_dict):
        segs = sample_prob_dict.get_segmentations(astype='list')
        assert isinstance(segs, list)


class TestEnforceProbDict:
    def test_enforce_prob_dict_passthrough(self):
        pd_obj = ProbabilityDictionary({"test": 0.5})
        result = enforce_prob_dict(pd_obj)
        assert result is pd_obj
    
    def test_enforce_prob_dict_from_dict(self):
        result = enforce_prob_dict({"test": 0.5})
        assert isinstance(result, ProbabilityDictionary)
    
    def test_enforce_prob_dict_from_list(self):
        result = enforce_prob_dict(["hello", "world"])
        assert isinstance(result, ProbabilityDictionary)
        assert result.dictionary["hello"] == 0.0
    
    def test_enforce_prob_dict_from_dataframe(self):
        df = pd.DataFrame({
            "segmentation": ["hello world"],
            "score": [0.5]
        })
        result = enforce_prob_dict(df)
        assert isinstance(result, ProbabilityDictionary)
```

#### 1.3 `tests/test_config.py` - Configuration

```python
import pytest
import tempfile
import json
from pathlib import Path
from hashformers.config import HashformersConfig

class TestHashformersConfig:
    def test_default_values(self):
        config = HashformersConfig()
        assert config.topk == 20
        assert config.steps == 13
        assert config.alpha == 0.222
        assert config.beta == 0.111
        assert config.device == 'cuda'
    
    def test_custom_values(self):
        config = HashformersConfig(topk=10, steps=5, device='cpu')
        assert config.topk == 10
        assert config.steps == 5
        assert config.device == 'cpu'
    
    def test_to_dict(self):
        config = HashformersConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d['topk'] == 20
    
    def test_from_dict(self):
        d = {'topk': 15, 'steps': 10}
        config = HashformersConfig.from_dict(d)
        assert config.topk == 15
        assert config.steps == 10
    
    def test_to_json_and_from_json(self):
        config = HashformersConfig(topk=25, alpha=0.5)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_json(f.name)
            loaded = HashformersConfig.from_json(f.name)
        
        assert loaded.topk == 25
        assert loaded.alpha == 0.5
```

#### 1.4 `tests/test_utils.py` - Utility Functions

```python
import pytest
import pandas as pd
from hashformers.utils.filtering import filter_top_k
from hashformers.utils.twitter import extract_hashtags, extract_hashtags_with_prefix

class TestFilterTopK:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'hashtag': ['abc', 'abc', 'abc', 'xyz', 'xyz'],
            'segmentation': ['a bc', 'ab c', 'abc', 'x yz', 'xyz'],
            'score': [0.1, 0.2, 0.3, 0.1, 0.5]
        })
    
    def test_filter_top_k_basic(self, sample_df):
        result = filter_top_k(sample_df, k=2, gold_field='hashtag')
        # Should have 2 per group = 4 total
        assert len(result) == 4
    
    def test_filter_top_k_keeps_lowest_scores(self, sample_df):
        result = filter_top_k(sample_df, k=1, gold_field='hashtag')
        # Should keep only the lowest score per group
        abc_row = result[result['hashtag'] == 'abc']
        assert abc_row['score'].values[0] == 0.1
    
    def test_filter_top_k_with_fill(self, sample_df):
        # Filter to k=3 with fill should pad groups with fewer entries
        result = filter_top_k(sample_df, k=3, gold_field='hashtag', fill=True)
        group_sizes = result.groupby('hashtag').size()
        assert all(group_sizes == 3)


class TestTwitterUtils:
    def test_extract_hashtags_basic(self):
        text = "Hello #world #python"
        result = extract_hashtags(text)
        assert result == ['world', 'python']
    
    def test_extract_hashtags_empty(self):
        text = "No hashtags here"
        result = extract_hashtags(text)
        assert result == []
    
    def test_extract_hashtags_special_chars(self):
        text = "#CamelCase #with_underscore #123number"
        result = extract_hashtags(text)
        assert 'CamelCase' in result
        assert 'with_underscore' in result
        assert '123number' in result
    
    def test_extract_hashtags_with_prefix(self):
        text = "Hello #world #python"
        result = extract_hashtags_with_prefix(text)
        assert result == ['#world', '#python']
```

#### 1.5 `tests/test_ensemble.py` - Ensemble Logic

```python
import pytest
import numpy as np
from hashformers.ensemble.top2_fusion import run_ensemble, Top2Ensembler

class TestRunEnsemble:
    def test_run_ensemble_basic(self):
        a_diff = np.array([0.1, 0.2])
        b_diff = np.array([0.3, 0.1])
        a_rank = np.array([0, 1])
        b_rank = np.array([1, 0])
        
        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=0.2, beta=0.1)
        assert len(result) == 2
    
    def test_run_ensemble_zero_weights(self):
        a_diff = np.array([0.1, 0.2])
        b_diff = np.array([0.3, 0.1])
        a_rank = np.array([0, 1])
        b_rank = np.array([1, 0])
        
        # With zero weights, delta = 0, so should use a_rank
        result = run_ensemble(a_diff, b_diff, a_rank, b_rank, alpha=0.0, beta=0.0)
        np.testing.assert_array_equal(result, a_rank)


class TestTop2Ensembler:
    def test_ensembler_init(self):
        ensembler = Top2Ensembler()
        assert ensembler is not None
```

### Phase 2: Integration Tests (GPU Optional)

#### 2.1 `tests/test_integrations.py`

```python
import pytest

class TestSpacyIntegration:
    @pytest.fixture
    def spacy_available(self):
        try:
            import spacy
            return True
        except ImportError:
            return False
    
    @pytest.mark.skipif(not spacy_available, reason="spaCy not installed")
    def test_spacy_component_registers(self, spacy_available):
        import spacy
        from hashformers.integrations import spacy_component
        # Component should be registered
        assert "hashformers" in spacy.Language.factories


class TestLangchainIntegration:
    @pytest.fixture
    def langchain_available(self):
        try:
            from langchain_core.documents import Document
            return True
        except ImportError:
            return False
    
    def test_transformer_init_without_langchain(self):
        from hashformers.integrations.langchain_integration import HashformersTransformer
        # Should not raise even without LangChain
        transformer = HashformersTransformer()
        assert transformer.segmenter_model == "gpt2"
```

### Phase 3: Test Infrastructure Improvements

1. **Add `pytest-mock` to test dependencies** for mocking model loading.
2. **Add `pytest-cov` for coverage reporting.**
3. **Create conftest.py** with shared fixtures.
4. **Set up CI coverage threshold** (target: 80% for core modules).

#### Update `requirements.txt` (test dependencies)

```
pytest>=7.0.0
pytest-lazy-fixture>=0.6.3
pytest-mock>=3.10.0
pytest-cov>=4.0.0
```

## Acceptance Criteria

- [ ] New test files created for each module listed above.
- [ ] All pure-logic functions have unit tests that run without GPU.
- [ ] `pytest` runs successfully on CPU-only CI/CD environment.
- [ ] Test coverage for core modules (`beamsearch`, `ensemble`, `utils`) reaches **≥80%**.
- [ ] Mocking strategy established for GPU-dependent model tests.
- [ ] `pytest-cov` integrated with coverage report generation.
- [ ] No existing tests are broken by changes.

## Estimated Effort

| Task | Time Estimate |
|------|---------------|
| Phase 1: Unit tests | 4-6 hours |
| Phase 2: Integration tests | 2-3 hours |
| Phase 3: Infrastructure | 1-2 hours |
| **Total** | **7-11 hours** |

