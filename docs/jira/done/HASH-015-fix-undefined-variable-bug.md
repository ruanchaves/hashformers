# HASH-015: Fix Potential Undefined Variable Bug in `update_probabilities`

## Summary
Fix bug where `current_batch_probs` may be undefined when `current_batch` is empty

## Priority
**Critical**

## Component
`beamsearch/algorithm.py`

## Type
Bug

---

## Description

In `update_probabilities` (lines 64-74), there's a potential bug where `current_batch_probs` can be undefined:

### Current Code
```python
for item in tree:
    current_batch = []
    for word in item:
        if word in prob_dict:
            continue
        else:
            current_batch.append(word)
    if current_batch:
        current_batch_probs = self.model.get_probs(current_batch)
    for idx, word in enumerate(current_batch):
        prob_dict[word] = current_batch_probs[idx]  # ← BUG: may be undefined!
return prob_dict
```

### Bug Scenario

1. First iteration: `current_batch` has items → `current_batch_probs` is set
2. Second iteration: `current_batch` is empty (all words cached)
3. The `if current_batch:` block is skipped
4. `current_batch_probs` **retains stale value from previous iteration**
5. If the code somehow references it, incorrect data is used

### Current Behavior
The loop `for idx, word in enumerate(current_batch)` doesn't execute when `current_batch` is empty, so the bug is currently silent. However:
- This is fragile and relies on implementation details
- Future refactoring could expose the bug
- Code is misleading about data flow

---

## Acceptance Criteria

- [ ] Add explicit handling when `current_batch` is empty
- [ ] Initialize `current_batch_probs` to prevent undefined state
- [ ] Add unit test covering the edge case
- [ ] Consider refactoring to make data flow clearer

---

## Suggested Implementation

### Option 1: Add `else: continue`
```python
for item in tree:
    current_batch = []
    for word in item:
        if word in prob_dict:
            continue
        else:
            current_batch.append(word)
    if current_batch:
        current_batch_probs = self.model.get_probs(current_batch)
        for idx, word in enumerate(current_batch):
            prob_dict[word] = current_batch_probs[idx]
    # else: nothing to do, skip to next item
return prob_dict
```

### Option 2: Initialize Variable
```python
for item in tree:
    current_batch = []
    current_batch_probs = []  # ← Initialize
    for word in item:
        if word in prob_dict:
            continue
        else:
            current_batch.append(word)
    if current_batch:
        current_batch_probs = self.model.get_probs(current_batch)
    for idx, word in enumerate(current_batch):
        prob_dict[word] = current_batch_probs[idx]
return prob_dict
```

### Option 3: Refactor for Clarity (Recommended)
```python
def update_probabilities(self, tree, prob_dict):
    for item in tree:
        # Collect words not yet scored
        new_words = [word for word in item if word not in prob_dict]
        
        if not new_words:
            continue
        
        # Score new words and add to dictionary
        scores = self.model.get_probs(new_words)
        for word, score in zip(new_words, scores):
            prob_dict[word] = score
    
    return prob_dict
```

---

## Test Case

```python
def test_update_probabilities_cached_words():
    """Test when all words are already in prob_dict."""
    beamsearch = Beamsearch(...)
    
    # Pre-populate cache
    prob_dict = {"hello": 0.5, "world": 0.3}
    tree = [["hello", "world"]]  # All words already cached
    
    # Should not crash or produce incorrect results
    result = beamsearch.update_probabilities(tree, prob_dict)
    
    assert result == prob_dict  # No changes expected
```

---

## Impact

| Area | Impact |
|------|--------|
| Correctness | Prevent potential silent bugs |
| Robustness | Handle edge cases explicitly |
| Maintainability | Clearer code intent |

---

## Labels
`bug`, `critical`, `algorithm`, `edge-case`
