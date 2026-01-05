# HASH-010: Add Type Hints to Core Functions in `algorithm.py`

## Summary
Add comprehensive type annotations to core beamsearch algorithm functions for IDE support and mypy compatibility

## Priority
**Medium**

## Component
`beamsearch/algorithm.py`

## Type
Code Quality

---

## Description

Core algorithm functions lack type hints, reducing IDE support and preventing static type checking:

### Current State (lines 33, 53, 126)
```python
def next_step(self, list_of_candidates):
    ...

def update_probabilities(self, tree, prob_dict):
    ...

def run(self, dataset, topk=20):
    ...
```

### Problems

1. **No IDE Autocomplete** — Editors can't infer parameter types
2. **No Static Checking** — `mypy` can't catch type errors
3. **Documentation Gap** — Unclear what types are expected
4. **Refactoring Risk** — Type mismatches caught at runtime

---

## Acceptance Criteria

- [ ] Add type hints to all public methods in `algorithm.py`
- [ ] Include return type annotations
- [ ] Add type hints to key data structures
- [ ] Ensure `mypy --strict` passes (or document exceptions)
- [ ] Update related modules (`reranker.py`, `top2_fusion.py`)

---

## Suggested Implementation

```python
from typing import List, Dict, Optional, Any

# Define type aliases for clarity
Hypothesis = str
Score = float
ProbabilityDict = Dict[Hypothesis, Score]
CandidateTree = List[List[Hypothesis]]

class Beamsearch(ModelLM):
    
    def next_step(self, list_of_candidates: List[str]) -> List[str]:
        """Generate next step candidates by inserting spaces."""
        ...
    
    def update_probabilities(
        self, 
        tree: CandidateTree, 
        prob_dict: ProbabilityDict
    ) -> ProbabilityDict:
        """Update probability dictionary with scores for new candidates."""
        ...
    
    def trim_tree(
        self, 
        tree: List[str], 
        prob_dict: ProbabilityDict, 
        topk: int
    ) -> List[str]:
        """Trim candidates to top-k per unique character sequence."""
        ...
    
    def run(
        self, 
        dataset: List[str], 
        topk: int = 20
    ) -> ProbabilityDict:
        """Run beamsearch segmentation on dataset."""
        ...
```

---

## Data Structure Types

```python
from dataclasses import dataclass
from typing import NamedTuple

class Node(NamedTuple):
    """Candidate node with hypothesis, characters, and score."""
    hypothesis: str
    characters: str
    score: float
```

---

## Verification

```bash
# Run mypy type checking
mypy src/hashformers/beamsearch/ --strict

# Check with pyright
pyright src/hashformers/beamsearch/
```

---

## Impact

| Area | Benefit |
|------|---------|
| IDE Support | Autocomplete and signature hints |
| Static Analysis | Catch bugs before runtime |
| Documentation | Self-documenting code |
| Refactoring | Safer code changes |

---

## Labels
`code-quality`, `medium-priority`, `type-hints`, `developer-experience`
