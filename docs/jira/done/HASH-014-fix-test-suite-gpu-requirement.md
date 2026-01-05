# HASH-014: Fix Test Suite GPU Requirement at Module Level

## Summary
Replace module-level GPU check exception with pytest skip decorators to enable CPU-only CI/CD

## Priority
**Medium**

## Component
`tests/test_segmenter.py`

## Type
Technical Debt / DevOps

---

## Description

The test file raises an exception at **import time** if no GPU is available:

### Current Code (lines 14-17)
```python
CUDA_IS_AVAILABLE = torch.cuda.is_available()

if not CUDA_IS_AVAILABLE:
    raise Exception("A GPU is required for these tests.")
```

### Problems

1. **CI/CD Blocked** — Cannot run ANY tests on CPU-only runners
2. **Import Failure** — Entire test module fails to load
3. **Mixed Tests Excluded** — CPU-compatible tests also blocked
4. **Poor User Experience** — Unclear error for contributors

---

## Acceptance Criteria

- [ ] Remove module-level exception
- [ ] Add `@pytest.mark.skipif` decorator for GPU-requiring tests
- [ ] Identify tests that can run on CPU (e.g., `test_twitter_text_matcher`, `test_regex_word_segmentation`)
- [ ] Add pytest marker `@pytest.mark.gpu` for GPU tests
- [ ] Configure CI to run CPU-compatible tests on all runners
- [ ] Document test categories in README or CONTRIBUTING

---

## Suggested Implementation

### Step 1: Remove Module-Level Check
```python
# tests/test_segmenter.py

import pytest
import torch

# Remove this block:
# if not torch.cuda.is_available():
#     raise Exception("A GPU is required for these tests.")
```

### Step 2: Add Skip Decorators
```python
import pytest

CUDA_AVAILABLE = torch.cuda.is_available()
GPU_SKIP_REASON = "GPU required for this test"

@pytest.mark.skipif(not CUDA_AVAILABLE, reason=GPU_SKIP_REASON)
def test_transformer_segmentation():
    """Test requiring GPU."""
    ...

def test_twitter_text_matcher():
    """Test that works on CPU."""
    ...

def test_regex_word_segmentation():
    """Test that works on CPU."""
    ...
```

### Step 3: Add Custom Marker (Optional)
```python
# conftest.py

import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
```

```python
# tests/test_segmenter.py

@pytest.mark.gpu
@pytest.mark.skipif(not CUDA_AVAILABLE, reason=GPU_SKIP_REASON)
def test_transformer_segmentation():
    ...
```

### Step 4: CI Configuration
```yaml
# .github/workflows/test.yml
jobs:
  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - run: pytest -m "not gpu"
  
  test-gpu:
    runs-on: [self-hosted, gpu]
    steps:
      - run: pytest
```

---

## Test Classification

| Test | GPU Required? |
|------|---------------|
| `test_transformer_segmentation` | Yes |
| `test_twitter_text_matcher` | No |
| `test_regex_word_segmentation` | No |
| `test_beamsearch_run` | Yes |
| `test_reranker` | Yes |

---

## Impact

| Area | Benefit |
|------|---------|
| CI/CD | Run tests on CPU runners |
| Contributor Experience | Clear test requirements |
| Test Coverage | Partial coverage without GPU |
| Faster Feedback | Quick CPU tests in PRs |

---

## Labels
`tech-debt`, `medium-priority`, `testing`, `ci-cd`
