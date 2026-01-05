# HASH-003: Create `HashformersConfig` Dataclass for Hyperparameters

## Summary
Replace hardcoded magic numbers with a centralized configuration dataclass for reproducibility and configurability

## Priority
**Critical**

## Component
`beamsearch/algorithm.py`, `segmenter/auto.py`, `beamsearch/top2_fusion.py`

## Type
Technical Debt / Architecture

---

## Description

Multiple hardcoded hyperparameters are scattered across the codebase without documentation:

| Parameter | Value | Location |
|-----------|-------|----------|
| `topk` | 20 | `algorithm.py:126` |
| `steps` | 13 | `auto.py:65` |
| `alpha` | 0.222 | `auto.py:66` |
| `beta` | 0.111 | `auto.py:67` |

### Current Code (auto.py:61-68)
```python
def segment(
        self,
        word_list,
        topk: int = 20,
        steps: int = 13,
        alpha: float = 0.222,
        beta: float = 0.111,
        use_reranker: bool = True,
```

### Problems

1. **No Documentation** — Values appear experimentally derived but lack explanation
2. **Scattered Configuration** — Same/similar values repeated in multiple files
3. **Reproducibility Issues** — Hard to understand what configuration was used
4. **Difficult Tuning** — No single place to adjust hyperparameters

---

## Acceptance Criteria

- [ ] Create `HashformersConfig` dataclass in dedicated config module
- [ ] Document the origin/rationale for each hyperparameter value
- [ ] Refactor all modules to use configuration object
- [ ] Support loading configuration from YAML/JSON files
- [ ] Add configuration serialization for reproducibility
- [ ] Maintain backward compatibility with existing defaults

---

## Suggested Implementation

```python
# src/hashformers/config.py
from dataclasses import dataclass, field
from typing import Optional
import json
import yaml

@dataclass
class HashformersConfig:
    """Configuration for Hashformers segmentation.
    
    Attributes:
        topk: Number of top candidates to retain per step (default: 20)
        steps: Maximum segmentation depth (default: 13)
        alpha: Beamsearch score weight (default: 0.222, empirically tuned)
        beta: Reranker score weight (default: 0.111, empirically tuned)
        gpu_batch_size: Batch size for GPU inference (default: 1000)
        device: Compute device (default: 'cuda')
    """
    topk: int = 20
    steps: int = 13
    alpha: float = 0.222  # TODO: Document empirical derivation
    beta: float = 0.111   # TODO: Document empirical derivation
    gpu_batch_size: int = 1000
    device: str = 'cuda'
    model_name_or_path: str = 'gpt2'
    model_type: str = 'gpt2'
    
    @classmethod
    def from_yaml(cls, path: str) -> 'HashformersConfig':
        with open(path, 'r') as f:
            return cls(**yaml.safe_load(f))
    
    def to_yaml(self, path: str) -> None:
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
```

---

## Impact

| Area | Benefit |
|------|---------|
| Reproducibility | Configuration saved with results |
| Maintainability | Single source of truth |
| User Experience | Easy hyperparameter tuning |
| Documentation | Self-documenting parameters |

---

## Labels
`tech-debt`, `critical`, `configuration`, `architecture`
