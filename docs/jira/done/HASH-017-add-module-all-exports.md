# HASH-017: Add `__all__` Exports to Package `__init__.py` Files

## Summary
Define explicit public API surface using `__all__` in package `__init__.py` files

## Priority
**Low**

## Component
All `__init__.py` files

## Type
Code Quality

---

## Description

Package `__init__.py` files lack `__all__` definitions, making the public API unclear.

## Acceptance Criteria

- [ ] Add `__all__` to main package `__init__.py`
- [ ] Add `__all__` to subpackage `__init__.py` files
- [ ] Include only intentionally public API

## Suggested Implementation

```python
# src/hashformers/__init__.py
__all__ = [
    "TransformerWordSegmenter",
    "Beamsearch", 
    "HashformersConfig",
]
```

## Labels
`code-quality`, `low-priority`, `api`, `documentation`
