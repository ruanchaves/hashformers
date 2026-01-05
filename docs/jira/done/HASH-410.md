Here is the Jira ticket to implement Top-k fusion support.

---

# HASH-410: Implement Top-k Fusion Support

## Summary

Generalize the existing Top-2 fusion ensemble logic to support Top-k candidates.

## Priority

**Medium**

## Component

`ensemble/top2_fusion.py`, `experiments/utils.py`

## Type

Feature Request

---

## Description

The current ensemble implementation (`Top2Ensembler`) is hardcoded to fuse exactly two candidates from the segmenter output. This limits the re-ranking capabilities to a pairwise comparison. We want to generalize this to support `k` candidates (e.g., top-5 or top-10), allowing for more robust re-ranking using the secondary model (reranker).

### Current Limitations

1. **Hardcoded Filtering**: The function `filter_and_project_scores` in `experiments/utils.py` explicitly calls `filter_top_k` with `k=2`.
2. **Pairwise Logic**: The function `calculate_diff_scores` in `experiments/utils.py` reshapes arrays to `(-1, 2)`, assuming exactly two candidates per hashtag.
3. **Scoring Function**: The current `run_ensemble` logic relies on calculating the difference (`delta`) between two scores. For , a simple pairwise difference is not defined. We need to implement a weighted fusion strategy.

---

## Acceptance Criteria

* [ ] Update `filter_and_project_scores` to accept a `k` parameter (defaulting to 2).
* [ ] Implement a generalized scoring strategy for  candidates (e.g., Weighted Sum Fusion).
* [ ] Refactor `Top2Ensembler` to a generic `TopKEnsembler` class.
* [ ] Ensure backward compatibility for the existing "Top-2" behavior (logic can remain for  or be approximated by the general case).
* [ ] Add unit tests for  and .

---

## Suggested Implementation

### 1. Generalize Utilities (`experiments/utils.py`)

**Update `filter_and_project_scores**`:
Allow passing `k` to the function.

```python
def filter_and_project_scores(a, b, k=2, characters_field="hashtag", segmentation_field="segmentation"):
    # ...
    # Replace hardcoded 2 with k
    models[0] = filter_top_k(models[0], k, fill=True) 
    # ...

```

**Add `calculate_weighted_scores**`:
Since `calculate_diff_scores` is specific to pairwise comparison (reshaping to `(-1, 2)`), implement a standard weighted fusion for the general case.

```python
def calculate_weighted_scores(a, b, alpha=1.0, beta=1.0, score_field="score"):
    # Merge a and b scores
    # Final Score = alpha * A.score + beta * B.score
    # Sort by Final Score
    pass

```

### 2. Refactor Ensembler (`ensemble/top2_fusion.py`)

Create `TopKEnsembler` or update `Top2Ensembler`.

```python
class TopKEnsembler:
    def __init__(self, k=2):
        self.k = k

    def run(self, segmenter_run, reranker_run, alpha=0.5, beta=0.5):
        # Implementation utilizing the new generalized utils
        pass

```

For backward compatibility, `Top2Ensembler` can inherit from `TopKEnsembler` or alias it with `k=2`.

---

## Impact

| Area | Benefit |
| --- | --- |
| Flexibility | Allows re-ranking deeper candidate lists (e.g., beam width > 2) |
| Performance | Potential accuracy gains by considering more candidates |
| Architecture | Decouples fusion logic from specific beam width |

---

## Labels

`feature`, `ensemble`, `medium-priority`