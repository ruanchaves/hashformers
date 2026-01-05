# HASH-013: Modernize Python 2/3 Casting Style in `Modeler`

## Summary
Replace legacy `(float)(expression)` casting syntax with modern Python style

## Priority
**Medium**

## Component
`evaluation/modeler.py`

## Type
Technical Debt

---

## Description

The `Modeler` class (lines 99-104) uses C-style casting syntax that is a Python 2 relic:

### Current Code
```python
(float)(self.p*100)
```

### Modern Python Style
```python
self.p * 100 / self.totals  # Let Python handle float division

# Or explicit if needed:
float(self.p * 100)
```

### Note
In Python 3, the `/` operator always returns a float, making explicit casting often unnecessary:
```python
>>> 5 / 2
2.5
>>> type(5 / 2)
<class 'float'>
```

---

## Acceptance Criteria

- [ ] Identify all instances of C-style casting `(type)(expr)`
- [ ] Replace with modern Python `type(expr)` or remove if redundant
- [ ] Verify numerical accuracy is maintained
- [ ] Update any related formatting/printing code

---

## Suggested Implementation

```python
# Before (modeler.py:99-104)
(float)(self.p*100)

# After - Option 1: Remove unnecessary cast
self.p * 100 / self.totals  # Python 3 division is float

# After - Option 2: Explicit modern cast
float(self.p * 100) / self.totals
```

---

## Search Pattern

Find all instances in codebase:
```bash
# Find C-style casts
grep -rn "(float)(" src/
grep -rn "(int)(" src/
grep -rn "(str)(" src/
```

---

## Impact

| Area | Benefit |
|------|---------|
| Readability | Standard Python idioms |
| Modernization | Python 3 style |
| Maintainability | Familiar patterns |

---

## Labels
`tech-debt`, `medium-priority`, `python3`, `code-modernization`
