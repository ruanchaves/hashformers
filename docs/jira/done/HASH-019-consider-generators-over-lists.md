# HASH-019: Consider Generators Over List Comprehensions for Memory Efficiency

## Summary
Profile and convert memory-intensive list comprehensions to generators where beneficial

## Priority
**Low**

## Component
`beamsearch/algorithm.py`

## Type
Performance

---

## Description

Some list comprehensions in `algorithm.py` (lines 43-50) could be generators for memory efficiency.

### Current Code
```python
candidates = [
    candidate_string[:pos] + ' ' + candidate_string[pos:]
    for pos in range(len(candidate_string))
]
```

## Acceptance Criteria

- [ ] Profile memory usage with current implementation
- [ ] Identify comprehensions that materialize large lists
- [ ] Convert to generators where memory savings are significant
- [ ] Benchmark to ensure no performance regression

## Notes

This is a minor optimization. Profile first before making changes.

## Labels
`performance`, `low-priority`, `optimization`, `memory`
