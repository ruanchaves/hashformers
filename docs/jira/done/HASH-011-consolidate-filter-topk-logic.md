# HASH-011: Consolidate Duplicated `filter_top_k` Logic

## Summary
Merge duplicate `filter_top_k` implementations into a single utility function following DRY principle

## Priority
**Medium**

## Component
`experiments/evaluation.py`, `beamsearch/data_structures.py`

## Type
Technical Debt

---

## Description

The `filter_top_k` logic is duplicated in two locations:

| Location | Lines |
|----------|-------|
| `experiments/evaluation.py` | 57-106 |
| `beamsearch/data_structures.py` | 68-121 |

### Problems

1. **DRY Violation** — Same logic maintained in two places
2. **Divergence Risk** — Fixes in one location may not reach the other
3. **Maintenance Burden** — Double the testing and review effort
4. **Inconsistency Risk** — Subtle behavioral differences may emerge

---

## Acceptance Criteria

- [ ] Identify the canonical implementation (likely `data_structures.py`)
- [ ] Consolidate shared logic into single utility module
- [ ] Update all callers to use the consolidated function
- [ ] Add comprehensive unit tests for the utility
- [ ] Remove duplicate implementation
- [ ] Document the function with examples

---

## Suggested Implementation

Create a shared utilities module:

```python
# src/hashformers/utils/filtering.py

from typing import List, Dict, TypeVar, Callable
from operator import itemgetter

T = TypeVar('T')

def filter_top_k(
    items: List[T],
    k: int,
    key: Callable[[T], float] = None,
    reverse: bool = False
) -> List[T]:
    """
    Filter items to top-k by score.
    
    Args:
        items: List of items to filter
        k: Number of top items to retain
        key: Optional function to extract score from item
        reverse: If True, return bottom-k instead
    
    Returns:
        Top-k items sorted by score
    
    Example:
        >>> items = [{'word': 'hello', 'score': 0.9}, {'word': 'world', 'score': 0.5}]
        >>> filter_top_k(items, k=1, key=lambda x: x['score'])
        [{'word': 'hello', 'score': 0.9}]
    """
    if key is None:
        key = lambda x: x
    
    sorted_items = sorted(items, key=key, reverse=not reverse)
    return sorted_items[:k]
```

Update callers:

```python
# evaluation.py
from hashformers.utils.filtering import filter_top_k

# data_structures.py  
from hashformers.utils.filtering import filter_top_k
```

---

## Migration Steps

1. Create `src/hashformers/utils/filtering.py`
2. Implement canonical `filter_top_k`
3. Add unit tests in `tests/test_filtering.py`
4. Update `evaluation.py` to use shared utility
5. Update `data_structures.py` to use shared utility
6. Remove inline implementations
7. Run full test suite

---

## Impact

| Area | Benefit |
|------|---------|
| Maintainability | Single point of change |
| Testing | Consolidated test coverage |
| Consistency | Guaranteed same behavior |
| Code Size | Reduced duplication |

---

## Labels
`tech-debt`, `medium-priority`, `dry`, `refactoring`
