---
name: Bug Report
about: Python 3.13 compatibility - lib2to3 removal breaks package import
title: "[Breaking] Python 3.13 Compatibility: Remove lib2to3 import (PEP 594)"
labels: bug, breaking-change, python-3.13, priority-critical
assignees: ''
---

## 🐛 Summary

**hashformers is incompatible with Python 3.13** due to an import from `lib2to3`, which was removed from the standard library in Python 3.13 per [PEP 594](https://peps.python.org/pep-0594/).

When importing hashformers on Python 3.13, users will encounter:
```
ModuleNotFoundError: No module named 'lib2to3'
```

This is a **critical failure** that completely prevents the package from being imported.

---

## 🔍 Audit Results

### Environment
- **Target Python Version:** 3.13+
- **Reference:** PEP 594 — Removing dead batteries from the standard library
- **Audit Date:** 2026-01-06

### Critical Failures

- [x] **`src/hashformers/segmenter/auto.py:1`** ✅ **FIXED**
  ```python
  from lib2to3.pytree import Base
  ```
  - **Impact:** Package fails to import entirely
  - **Severity:** 🔴 Critical
  - **Note:** The imported `Base` class is **never used** in the file — this is a dead import
  - **Resolution:** Line removed on 2026-01-06

### Import Chain Affected

```
hashformers/__init__.py
  └─> hashformers/segmenter/__init__.py
        └─> hashformers/segmenter/segmenter.py
              └─> (no issue)
  └─> hashformers/segmenter/auto.py  ❌ FAILS HERE
        └─> from lib2to3.pytree import Base
```

Because `hashformers/__init__.py` imports from `segmenter/auto.py`, **the entire package is broken**.

---

## ✅ No Issues Found

The following PEP 594 removed modules were **not detected** in the codebase:

| Module | Status |
|--------|--------|
| `aifc` | ✅ Not used |
| `audioop` | ✅ Not used |
| `cgi` / `cgitb` | ✅ Not used |
| `chunk` | ✅ Not used |
| `crypt` | ✅ Not used |
| `imghdr` | ✅ Not used |
| `mailcap` | ✅ Not used |
| `msilib` | ✅ Not used |
| `nis` | ✅ Not used |
| `nntplib` | ✅ Not used |
| `ossaudiodev` | ✅ Not used |
| `pipes` | ✅ Not used |
| `sndhdr` | ✅ Not used |
| `spwd` | ✅ Not used |
| `sunau` | ✅ Not used |
| `telnetlib` | ✅ Not used |
| `uu` | ✅ Not used |
| `xdrlib` | ✅ Not used |
| `distutils` | ✅ Not used |
| `asyncore` / `asynchat` | ✅ Not used |

**Other checks passed:**
- ✅ No deprecated `collections` ABCs (use `collections.abc`)
- ✅ No deprecated `unittest` assert methods
- ✅ No `@asyncio.coroutine` decorators
- ✅ No deprecated SSL protocols
- ✅ No deprecated `urllib.parse` functions
- ✅ No `pkg_resources` usage

---

## 🔧 Recommended Fixes

### Fix 1: Remove Dead Import (Preferred)

The `Base` class from `lib2to3.pytree` is imported but **never used** in `auto.py`. Simply delete the import:

```diff
- from lib2to3.pytree import Base
  from hashformers.segmenter import (
      BaseWordSegmenter
  )
```

**File:** `src/hashformers/segmenter/auto.py`  
**Line:** 1  
**Action:** Delete line 1 entirely

### Fix 2: Verify No Actual Usage

Before removing, confirm `Base` is not used elsewhere:

```bash
grep -r "lib2to3" src/
grep -r "\.Base\b" src/hashformers/segmenter/auto.py
```

The audit confirmed:
- `Base` is only referenced via import
- No methods or inheritance from `lib2to3.pytree.Base` are present
- Other `Base` references in the codebase refer to `BaseWordSegmenter` and `BaseSegmenter` (internal classes)

---

## ⚠️ Deprecation Warnings (Non-Breaking) — ✅ FIXED

These patterns were deprecated but have been updated to modern Python 3.9+ syntax:

### `typing.List`, `typing.Dict`, etc.

Since Python 3.9, built-in generics are preferred over `typing` module equivalents:

| File | Line | Status | Change |
|------|------|--------|--------|
| `src/hashformers/segmenter/segmenter.py` | 2 | ✅ Fixed | `List` → `list` |
| `src/hashformers/segmenter/regex_segmenter.py` | 2 | ✅ Fixed | `List` → `list` |
| `src/hashformers/segmenter/data_structures.py` | 1 | ✅ Fixed | `List` → `list` |

**Example fix:**
```diff
- from typing import List, Any
+ from typing import Any  # Keep Any, Union if needed

- def method(self, items: List[str]) -> List[str]:
+ def method(self, items: list[str]) -> list[str]:
```

**Note:** This is a style preference for Python 3.9+. The `typing` imports still work but generate deprecation warnings in some linters.

---

## 📋 Checklist

- [x] Remove `from lib2to3.pytree import Base` from `auto.py` ✅ **DONE**
- [ ] Run test suite to confirm no regressions
- [ ] Test import on Python 3.13
- [x] (Optional) Update `typing.List` → `list` throughout codebase ✅ **DONE**
- [ ] (Optional) Add Python 3.13 to CI test matrix
- [ ] Update `setup.py` or `pyproject.toml` with `python_requires=">=3.9,<3.14"` or similar

---

## 🧪 Testing

After applying the fix:

```bash
# Install Python 3.13
pyenv install 3.13.0
pyenv local 3.13.0

# Test import
python -c "from hashformers import TransformerWordSegmenter; print('✅ Import successful')"

# Run test suite
pytest tests/
```

---

## 📚 References

- [PEP 594 – Removing dead batteries from the standard library](https://peps.python.org/pep-0594/)
- [Python 3.13 Release Notes](https://docs.python.org/3.13/whatsnew/3.13.html)
- [lib2to3 Removal Notice](https://docs.python.org/3.11/library/2to3.html#module-lib2to3)

