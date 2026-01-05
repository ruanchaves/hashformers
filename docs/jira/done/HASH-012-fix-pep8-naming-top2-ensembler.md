# HASH-012: Fix PEP8 Naming Violation in `Top2_Ensembler` Class

## Summary
Rename `Top2_Ensembler` to `Top2Ensembler` to follow PEP8 class naming conventions

## Priority
**Medium**

## Component
`beamsearch/top2_fusion.py`

## Type
Code Quality

---

## Description

The class `Top2_Ensembler` (line 74) uses an underscore in the class name, violating PEP8 naming conventions.

### Current
```python
class Top2_Ensembler:
    ...
```

### PEP8 Convention
Class names should use **CapWords** (CamelCase) convention:
> "Class names should normally use the CapWords convention."
> — [PEP 8 - Naming Conventions](https://peps.python.org/pep-0008/#class-names)

---

## Acceptance Criteria

- [ ] Rename class from `Top2_Ensembler` to `Top2Ensembler`
- [ ] Update all imports and references across codebase
- [ ] Add backwards-compatible alias (optional, if public API)
- [ ] Update documentation and docstrings
- [ ] Run test suite to verify no regressions

---

## Suggested Implementation

### Step 1: Rename Class
```python
# top2_fusion.py
class Top2Ensembler:  # Removed underscore
    """Ensemble two ranking models using top-2 fusion."""
    ...
```

### Step 2: Add Backwards Compatibility (Optional)
```python
# For public API backwards compatibility
Top2_Ensembler = Top2Ensembler  # Deprecated alias

import warnings

def __getattr__(name):
    if name == 'Top2_Ensembler':
        warnings.warn(
            "Top2_Ensembler is deprecated, use Top2Ensembler",
            DeprecationWarning,
            stacklevel=2
        )
        return Top2Ensembler
    raise AttributeError(f"module has no attribute {name}")
```

### Step 3: Update Imports
```bash
# Find all references
grep -rn "Top2_Ensembler" src/ tests/
```

---

## Files to Update

| File | Change |
|------|--------|
| `beamsearch/top2_fusion.py` | Rename class definition |
| `segmenter/segmenter.py` | Update import (if applicable) |
| `segmenter/auto.py` | Update import (if applicable) |
| `tests/test_*.py` | Update test references |

---

## Impact

| Area | Benefit |
|------|---------|
| Consistency | Follows Python standards |
| Linting | No PEP8 warnings |
| Readability | Standard naming pattern |

---

## Labels
`code-quality`, `medium-priority`, `pep8`, `naming`
