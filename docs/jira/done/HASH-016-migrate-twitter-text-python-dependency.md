# HASH-016: Migrate Away from Deprecated `twitter-text-python` Dependency

## Summary
Replace deprecated `twitter-text-python` (last updated 2014) with maintained alternative or internal implementation

## Priority
**Low** (Roadmap)

## Component
`requirements.txt`, `segmenter/`

## Type
Technical Debt / Security

---

## Description

The `twitter-text-python` (`ttp`) package in `requirements.txt` was last updated in **2014** — over a decade ago.

### Current Dependency
```
twitter-text-python
```

### Concerns

1. **Unmaintained** — No updates in 10+ years
2. **Compatibility Risk** — May not support modern Python features
3. **Security Risk** — No security patches
4. **Twitter API Changes** — Hashtag/mention rules may have changed
5. **Dependency Conflict** — Old package may conflict with modern deps

---

## Acceptance Criteria

- [ ] Audit current usage of `twitter-text-python` in codebase
- [ ] Evaluate alternatives (see options below)
- [ ] Implement migration or internal replacement
- [ ] Update requirements and documentation
- [ ] Verify feature parity with current functionality

---

## Migration Options

### Option 1: Use `twitter-text` Official Package
```bash
pip install twitter-text
```

The official Twitter-maintained package (though also note Twitter → X transition may affect maintenance).

### Option 2: Internal Implementation
If only hashtag extraction is needed:

```python
import re

HASHTAG_PATTERN = re.compile(r'#(\w+)', re.UNICODE)

def extract_hashtags(text: str) -> list[str]:
    """Extract hashtags from text.
    
    Args:
        text: Input text containing hashtags
    
    Returns:
        List of hashtag strings without # prefix
    
    Example:
        >>> extract_hashtags("Hello #world #python")
        ['world', 'python']
    """
    return HASHTAG_PATTERN.findall(text)
```

### Option 3: Use `tweepy` Utilities
If already using `tweepy` for Twitter API access, it has text parsing utilities.

---

## Usage Audit

Search for current usage:
```bash
grep -rn "twitter" src/
grep -rn "ttp" src/
grep -rn "extract_hashtag" src/
```

Common usages to check:
- Hashtag extraction
- Mention extraction  
- URL extraction
- Text entity parsing

---

## Migration Steps

1. **Audit**: Identify all `ttp` usages
2. **Evaluate**: Determine minimal functionality needed
3. **Implement**: Create `hashformers/utils/twitter.py` if simple
4. **Test**: Verify parity with existing behavior
5. **Migrate**: Update imports across codebase
6. **Remove**: Delete `twitter-text-python` from requirements
7. **Document**: Update any relevant documentation

---

## Impact

| Area | Benefit |
|------|---------|
| Security | Remove unmaintained dependency |
| Compatibility | Modern Python support |
| Maintenance | Reduce external dependencies |
| Control | Own the extraction logic |

---

## Labels
`tech-debt`, `low-priority`, `dependencies`, `roadmap`, `security`
