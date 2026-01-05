# HASH-009: Pre-compile Regex Patterns for Performance

## Summary
Move regex compilation outside of loops to avoid redundant compilation overhead

## Priority
**High**

## Component
`segmenter/segmenter.py`, `beamsearch/algorithm.py`

## Type
Performance

---

## Description

Multiple regex patterns are compiled inside loops, causing repeated compilation overhead:

### Location 1: `segmenter.py:302-305`
Regex compiled per tweet in `build_hashtag_container`:

```python
# Current (inside loop)
for tweet in tweets:
    pattern = re.compile(r"...")  # Compiled every iteration!
```

### Location 2: `algorithm.py:43-50`
Double-space regex compiled per candidate:

```python
def next_step(self, list_of_candidates):
    output = []
    for candidate_string in list_of_candidates:
        candidates = [...]
        # Regex compiled on EVERY candidate
        candidates = list(filter(lambda x: not re.findall(".*?(?=\s{2})", x), candidates))
        output.extend(candidates)
```

### Performance Impact

- `re.compile()` has overhead even with caching
- `re.findall()` compiles pattern each time
- For 1000 tweets with 100 candidates each = 100,000 compilations

---

## Acceptance Criteria

- [ ] Identify all regex patterns compiled inside loops
- [ ] Move pattern compilation to module level or class initialization
- [ ] Use compiled pattern objects in hot paths
- [ ] Benchmark before/after performance
- [ ] Target: 2-3ms savings per tweet

---

## Suggested Implementation

### algorithm.py

```python
import re

# Module-level pre-compiled pattern
DOUBLE_SPACE_PATTERN = re.compile(r".*?(?=\s{2})")

class Beamsearch(ModelLM):
    
    def next_step(self, list_of_candidates):
        output = []
        for candidate_string in list_of_candidates:
            candidates = [
                candidate_string[:pos] + ' ' + candidate_string[pos:]
                if pos else candidate_string 
                for pos in range(len(candidate_string))
            ]
            # Use pre-compiled pattern
            candidates = [x for x in candidates if not DOUBLE_SPACE_PATTERN.findall(x)]
            output.extend(candidates)
        return output
```

### segmenter.py

```python
# Module-level or class-level
HASHTAG_PATTERN = re.compile(r"#\w+")  # Adjust pattern as needed

def build_hashtag_container(self, tweets):
    for tweet in tweets:
        matches = HASHTAG_PATTERN.findall(tweet)
        # ...
```

---

## Impact

| Metric | Improvement |
|--------|-------------|
| Per-Tweet Latency | 2-3ms reduction |
| Large Batch Processing | Significant cumulative savings |
| Memory Churn | Reduced object creation |

---

## Labels
`performance`, `high-priority`, `regex`, `optimization`
