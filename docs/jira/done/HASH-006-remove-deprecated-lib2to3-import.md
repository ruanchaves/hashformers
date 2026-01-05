# HASH-006: Remove Unused Deprecated `lib2to3` Import

## Summary
Remove unused `from lib2to3.pytree import Base` import to prevent Python 3.13+ breakage

## Priority
**High**

## Component
`segmenter/auto.py`

## Type
Technical Debt

---

## Description

Line 1 of `auto.py` contains an unused import:

```python
from lib2to3.pytree import Base
```

### Problems

1. **Unused Code** — The `Base` class is never used in the module
2. **Deprecated Module** — `lib2to3` is deprecated as of Python 3.11
3. **Removal Planned** — Scheduled for removal in Python 3.13+
4. **Import Overhead** — Unnecessary module loading

### Python Deprecation Notice
```
DeprecationWarning: lib2to3 package is deprecated and may not be 
able to parse Python 3.10+ code
```

---

## Acceptance Criteria

- [ ] Remove the unused import from `auto.py`
- [ ] Verify no other files import from `lib2to3`
- [ ] Run test suite to confirm no regressions
- [ ] Update minimum Python version requirements if needed

---

## Implementation

Simply delete line 1 from `segmenter/auto.py`:

```diff
- from lib2to3.pytree import Base
```

---

## Testing

```bash
# Verify no lib2to3 usage across codebase
grep -r "lib2to3" src/

# Run test suite
pytest tests/
```

---

## Impact

| Area | Benefit |
|------|---------|
| Compatibility | Python 3.13+ ready |
| Code Cleanliness | Remove dead code |
| Import Time | Minor improvement |

---

## Labels
`tech-debt`, `high-priority`, `python-compatibility`, `cleanup`
