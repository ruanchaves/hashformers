# HASH-008: Refactor `Modeler` Class-Level Mutable State to Instance Attributes

## Summary
Move class-level mutable attributes to instance attributes in `__init__` to ensure thread-safety

## Priority
**High**

## Component
`evaluation/modeler.py`

## Type
Architecture / Bug

---

## Description

The `Modeler` class (lines 17-28) uses class-level mutable attributes:

```python
class Modeler(object):
    
    hashtagSegmentor = None
    t = 0
    totals = 0
    totalh = 0
    p = 0
    r = 0
    n = 0
    
    modelerParams = {}
```

### Thread-Safety Hazard

Class-level attributes are **shared across all instances**. This creates race conditions when:

1. Multiple threads evaluate simultaneously
2. Multiple instances are created without proper isolation
3. The `reset()` method is called concurrently

### Bug Scenario
```python
# Thread 1
modeler1 = Modeler()
modeler1.evaluate(data1)  # Sets totals = 100

# Thread 2 (simultaneously)
modeler2 = Modeler()
modeler2.evaluate(data2)  # Corrupts totals!

# Both threads share the same class attributes
print(Modeler.totals)  # Unpredictable value
```

---

## Acceptance Criteria

- [ ] Move all mutable state to instance attributes in `__init__`
- [ ] Keep class-level constants if any (immutable only)
- [ ] Update `reset()` method to work with instance state
- [ ] Add tests verifying instance isolation
- [ ] Consider making `Modeler` thread-safe with locks if needed

---

## Suggested Implementation

```python
class Modeler:
    """Evaluation metrics modeler with per-instance state."""
    
    def __init__(self):
        self.hashtagSegmentor = None
        self.t = 0
        self.totals = 0
        self.totalh = 0
        self.p = 0
        self.r = 0
        self.n = 0
        self.modelerParams = {}
    
    def reset(self):
        """Reset all metrics to initial state."""
        self.t = 0
        self.totals = 0
        self.totalh = 0
        self.p = 0
        self.r = 0
        self.n = 0
```

---

## Testing

Add isolation test:

```python
def test_modeler_instance_isolation():
    modeler1 = Modeler()
    modeler2 = Modeler()
    
    modeler1.totals = 100
    modeler2.totals = 200
    
    assert modeler1.totals == 100  # Should not be affected by modeler2
    assert modeler2.totals == 200
```

---

## Impact

| Area | Benefit |
|------|---------|
| Thread Safety | Concurrent evaluation support |
| Instance Isolation | Independent modeler instances |
| Predictability | Deterministic behavior |
| Testing | Easier to mock and test |

---

## Labels
`architecture`, `high-priority`, `thread-safety`, `bug`
