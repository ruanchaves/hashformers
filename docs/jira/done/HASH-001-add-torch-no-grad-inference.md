# HASH-001: Add `torch.no_grad()` Context During Inference

## Summary
Wrap inference code in `torch.no_grad()` context to prevent unnecessary computation graph building

## Priority
**Critical**

## Component
`beamsearch/minicons_lm.py`

## Type
Performance / Bug

---

## Description

The `get_probs` method in `minicons_lm.py` (lines 13-18) performs model inference without disabling gradient computation. This causes PyTorch to build computation graphs during inference, which:

1. **Wastes GPU memory** — gradient tensors are allocated unnecessarily
2. **Increases OOM risk** — memory consumption grows significantly
3. **Slows down inference** — additional overhead from graph tracking

### Current Code
```python
def get_probs(self, list_of_candidates):
    probs = []
    dl = DataLoader(list_of_candidates, batch_size=self.gpu_batch_size)
    for batch in dl:
        probs.extend(self.get_batch_scores(batch))
    return probs
```

### Expected Behavior
Inference should be performed with gradient computation disabled to optimize memory usage and performance.

---

## Acceptance Criteria

- [ ] Wrap the inference loop in `with torch.no_grad():` context manager
- [ ] Verify no functional changes to scoring output
- [ ] Confirm memory usage reduction (~30-50%)
- [ ] Update any related scoring methods that perform inference

---

## Suggested Implementation

```python
import torch

def get_probs(self, list_of_candidates):
    probs = []
    dl = DataLoader(list_of_candidates, batch_size=self.gpu_batch_size)
    with torch.no_grad():  # ← Add this wrapper
        for batch in dl:
            probs.extend(self.get_batch_scores(batch))
    return probs
```

---

## Impact

| Metric | Improvement |
|--------|-------------|
| Memory Usage | 30-50% reduction |
| OOM Prevention | High impact |
| Inference Speed | Moderate improvement |

---

## Labels
`performance`, `critical`, `inference`, `pytorch`
