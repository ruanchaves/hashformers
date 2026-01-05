# HASH-007: Fix Mutable Default Arguments in `segment` Method

## Summary
Replace mutable default arguments (`dict = {}`) with `None` and internal defaulting to prevent shared state bugs

## Priority
**High**

## Component
`segmenter/segmenter.py`

## Type
Bug / Technical Debt

---

## Description

The `segment` method in `segmenter.py` (lines 81-91) uses mutable default arguments:

```python
def segment(
        self,
        word_list: List[str],
        segmenter_run: Any = None,
        preprocessing_kwargs: dict = {},   # ← Mutable default
        segmenter_kwargs: dict = {},        # ← Mutable default
        ensembler_kwargs: dict = {},        # ← Mutable default
        reranker_kwargs: dict = {},         # ← Mutable default
        use_reranker: bool = True,
        use_ensembler: bool = True,
        return_ranks: bool = False) -> Any:
```

### Why This Is Dangerous

Mutable default arguments are evaluated **once** at function definition time, not at each call. This means:

1. **Shared State** — All calls share the same dictionary object
2. **Silent Bugs** — Modifications persist across calls
3. **Non-Deterministic Behavior** — Results depend on call history

### Example Bug Scenario
```python
# First call modifies the internal dict
segmenter.segment(words, preprocessing_kwargs={})
# preprocessing_kwargs now contains modifications

# Second call unexpectedly inherits modifications!
segmenter.segment(other_words)  # May have stale state
```

---

## Acceptance Criteria

- [ ] Change all `dict = {}` defaults to `dict = None`
- [ ] Add internal defaulting logic at start of method
- [ ] Apply fix to all affected methods across codebase
- [ ] Add unit tests verifying isolation between calls
- [ ] Consider using `field(default_factory=dict)` if using dataclasses

---

## Suggested Implementation

```python
def segment(
        self,
        word_list: List[str],
        segmenter_run: Any = None,
        preprocessing_kwargs: dict = None,  # ← Changed to None
        segmenter_kwargs: dict = None,
        ensembler_kwargs: dict = None,
        reranker_kwargs: dict = None,
        use_reranker: bool = True,
        use_ensembler: bool = True,
        return_ranks: bool = False) -> Any:
    
    # Internal defaulting
    if preprocessing_kwargs is None:
        preprocessing_kwargs = {}
    if segmenter_kwargs is None:
        segmenter_kwargs = {}
    if ensembler_kwargs is None:
        ensembler_kwargs = {}
    if reranker_kwargs is None:
        reranker_kwargs = {}
    
    # ... rest of method
```

---

## Files to Check

Search for other instances across the codebase:
```bash
grep -rn "def.*= {}" src/
grep -rn "def.*= \[\]" src/
```

---

## Impact

| Area | Benefit |
|------|---------|
| Correctness | Prevent shared state bugs |
| Predictability | Deterministic behavior |
| Thread Safety | Calls don't interfere |

---

## Labels
`bug`, `high-priority`, `python-antipattern`, `tech-debt`
