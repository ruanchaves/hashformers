# HASH-004: Pin Dependency Versions in `requirements.txt`

## Summary
Add version pins to all dependencies to prevent breaking changes from upstream packages

## Priority
**Critical**

## Component
`requirements.txt`

## Type
Technical Debt / DevOps

---

## Description

The current `requirements.txt` contains unpinned dependencies:

```
minicons
twitter-text-python
ekphrasis
pandas
```

### Problems

1. **Reproducibility** — Different installations may get different package versions
2. **Breaking Changes** — Upstream updates can break functionality
3. **CI/CD Flakiness** — Builds may fail unexpectedly due to dependency updates
4. **Debugging Difficulty** — Hard to reproduce issues across environments

---

## Acceptance Criteria

- [ ] Audit current working versions of all dependencies
- [ ] Add minimum version pins with `>=` operator
- [ ] Consider adding maximum version pins for critical packages
- [ ] Create `requirements-dev.txt` for development dependencies
- [ ] Document Python version requirements
- [ ] Test installation in fresh environment

---

## Suggested Implementation

```
# requirements.txt
minicons>=0.2.0
twitter-text-python>=1.1.0  # Note: Last updated 2014, consider alternatives
ekphrasis>=0.5.1
pandas>=1.3.0
torch>=1.9.0
transformers>=4.10.0
```

Consider also adding a `pyproject.toml` for modern Python packaging:

```toml
[project]
name = "hashformers"
requires-python = ">=3.8"
dependencies = [
    "minicons>=0.2.0",
    "pandas>=1.3.0",
    "torch>=1.9.0",
    "transformers>=4.10.0",
]
```

---

## Additional Notes

> ⚠️ **Warning**: The `twitter-text-python` package was last updated in 2014. Consider:
> - Migrating to `twitter-text` (official Twitter package)
> - Implementing minimal hashtag extraction internally
> - See ticket HASH-016 for deprecation migration

---

## Impact

| Area | Benefit |
|------|---------|
| Reproducibility | Consistent installations |
| Stability | Protection from breaking changes |
| CI/CD | Reliable builds |
| Debugging | Easier issue reproduction |

---

## Labels
`tech-debt`, `critical`, `dependencies`, `devops`
