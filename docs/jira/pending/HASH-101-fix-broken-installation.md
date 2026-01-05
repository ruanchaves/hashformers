# HASH-101: Fix broken installation by syncing `setup.py` with `requirements.txt`

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Type**    | Bug                                                       |
| **Priority**| Highest                                                   |
| **Epic**    | Epic 1: Packaging & Positioning (Immediate Fixes)         |
| **Files**   | `setup.py`, `requirements.txt`                            |

## Description

The current `setup.py` is missing critical dependencies found in `requirements.txt`. A user installing via `pip install .` will experience runtime crashes.

### Current State

**`setup.py` install_requires:**
```python
install_requires=[
    "minicons",
    "twitter-text-python",
    "pandas"
]
```

**`requirements.txt` (full list):**
```
minicons>=0.2.0
twitter-text-python>=1.1.0
ekphrasis>=0.5.1
pandas>=1.3.0
torch>=1.9.0
transformers>=4.10.0
```

> [!CAUTION]
> Missing `torch` and `transformers` will cause immediate `ModuleNotFoundError` on import.

## Tasks

1. Open `setup.py`.
2. Update the `install_requires` list to include all core libraries currently listed in `requirements.txt`:
   - `torch>=1.9.0`
   - `transformers>=4.10.0`
   - `ekphrasis>=0.5.1` (verify if this is still needed by checking imports, otherwise remove from both files).
   - Update existing dependencies with version constraints:
     - `minicons>=0.2.0`
     - `twitter-text-python>=1.1.0`
     - `pandas>=1.3.0`
3. Add `python_requires='>=3.7'` to ensure compatibility.
4. **Verification:** Create a fresh virtual environment, run `pip install .`, and verify `import hashformers` works without `ModuleNotFoundError`.

## Acceptance Criteria

- [ ] `pip install .` installs `torch` and `transformers` automatically.
- [ ] All version constraints in `setup.py` match `requirements.txt`.
- [ ] `python_requires='>=3.7'` is present in `setup.py`.
- [ ] CI/CD pipeline (if exists) passes installation checks.
- [ ] Fresh virtualenv install + `import hashformers` succeeds without errors.
