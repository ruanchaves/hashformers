# HASH-018: Document Security Considerations for Hub Model Loading

## Summary
Add security documentation and optional hash verification for arbitrary Hugging Face Hub model loading

## Priority
**Low**

## Component
`beamsearch/algorithm.py`, `beamsearch/reranker.py`

## Type
Security

---

## Description

The `Beamsearch.__init__` and `Reranker.__init__` methods load arbitrary models from Hugging Face Hub without signature verification.

### Security Risk

- Models can contain arbitrary code (pickle files)
- Supply-chain attacks are possible
- No verification of model authenticity

## Acceptance Criteria

- [ ] Document security considerations in README
- [ ] Add optional `model_hash` parameter for verification
- [ ] Log warnings when loading unverified models
- [ ] Consider using `trust_remote_code=False` by default

## Suggested Implementation

```python
def __init__(self, model_name_or_path: str, verify_hash: str = None):
    if verify_hash:
        # Verify model files match expected hash
        pass
    else:
        logger.warning(
            f"Loading unverified model: {model_name_or_path}. "
            "Consider using verify_hash for security."
        )
```

## Labels
`security`, `low-priority`, `documentation`, `roadmap`
