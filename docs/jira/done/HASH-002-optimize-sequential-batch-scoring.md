# HASH-002: Optimize Sequential Batch Scoring in `update_probabilities`

## Summary
Refactor `update_probabilities` to flatten candidates for single mega-batch scoring instead of sequential processing

## Priority
**Critical**

## Component
`beamsearch/algorithm.py`

## Type
Performance

---

## Description

The `update_probabilities` method in `algorithm.py` (lines 64-75) processes tree items sequentially in a `for item in tree` loop. This pattern prevents GPU parallelization and loses significant throughput gains.

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
        prob_dict[word] = current_batch_probs[idx]
return prob_dict
```

### Problems

1. **Sequential Processing** — Each tree item is processed individually
2. **Multiple GPU Calls** — Separate `get_probs()` calls per batch
3. **Underutilized GPU** — Small batches don't saturate GPU compute

---

## Acceptance Criteria

- [ ] Flatten all unique candidates across tree items into single collection
- [ ] Score all candidates in one `get_probs()` call (or minimal batched calls)
- [ ] Redistribute scores back to probability dictionary
- [ ] Maintain functional equivalence with existing behavior
- [ ] Achieve 40-60% latency reduction in benchmarks

---

## Suggested Implementation

```python
def update_probabilities(self, tree, prob_dict):
    # Collect all unique words not yet in prob_dict
    all_candidates = set()
    for item in tree:
        for word in item:
            if word not in prob_dict:
                all_candidates.add(word)
    
    # Score all candidates in single mega-batch
    if all_candidates:
        candidates_list = list(all_candidates)
        all_probs = self.model.get_probs(candidates_list)
        for word, prob in zip(candidates_list, all_probs):
            prob_dict[word] = prob
    
    return prob_dict
```

---

## Impact

| Metric | Improvement |
|--------|-------------|
| Latency | 40-60% reduction |
| GPU Utilization | Significantly improved |
| Throughput | Major increase |

---

## Labels
`performance`, `critical`, `gpu-optimization`, `algorithm`
