# HASH-403: Update Defaults to "Fast Mode"

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Type**    | Task                                                      |
| **Priority**| High                                                      |
| **Epic**    | Polish & Marketing                                        |
| **Files**   | `src/hashformers/beamsearch/algorithm.py`                 |

## Goal

Ensure users get the best performance out of the box by enabling optimized defaults.

## Description

Update the default settings in `Beamsearch` to use token-based operations and KV-caching by default. This will provide significant performance improvements for users without requiring manual configuration.

## Current State

```python
def __init__(
    self,
    model_name_or_path: str = "gpt2", 
    model_type: str = "gpt2", 
    device: str = 'cuda', 
    gpu_batch_size: int = 1000,
    use_token_mode: bool = False,  # Currently disabled
    use_kv_cache: bool = False     # Currently disabled
):
```

## Tasks

1. In `algorithm.py`, change defaults:
   ```python
   use_token_mode: bool = True,
   use_kv_cache: bool = True
   ```

2. Add a fallback check: if the model structure doesn't support KV-caching, log a warning and fallback to non-cached mode gracefully.

3. Update any relevant documentation to reflect the new defaults.

4. Add a `legacy_mode` parameter for users who need the old behavior.

## Fallback Implementation

```python
def __init__(self, ..., use_token_mode: bool = True, use_kv_cache: bool = True):
    # ... existing code ...
    
    # Validate KV-cache support
    if self.use_kv_cache:
        if not self._model_supports_kv_cache():
            logger.warning(
                f"Model '{model_name_or_path}' does not support KV-caching. "
                "Falling back to standard mode."
            )
            self.use_kv_cache = False
```

## Acceptance Criteria

- [ ] `use_token_mode=True` is the default
- [ ] `use_kv_cache=True` is the default
- [ ] Graceful fallback with warning when model doesn't support caching
- [ ] Existing tests continue to pass
- [ ] Performance benchmarks show improvement for default configuration
- [ ] Documentation updated to reflect new defaults
