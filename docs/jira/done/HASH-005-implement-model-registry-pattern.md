# HASH-005: Implement Model Registry Pattern for `ModelLM` Factory

## Summary
Replace string-based type dispatch with registry pattern to enable extensibility without modifying core code

## Priority
**High**

## Component
`beamsearch/model_lm.py`

## Type
Architecture

---

## Description

The `ModelLM` factory (lines 24-39) uses string-based type dispatch to instantiate model classes. This violates the Open-Closed Principle — adding new model types requires modifying the core factory code.

### Current Pattern (Assumed)
```python
def create_model(model_type: str):
    if model_type == "gpt2":
        return GPT2LM(...)
    elif model_type == "bert":
        return BertLM(...)
    elif model_type == "minicons":
        return MiniconsLM(...)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### Problems

1. **Violates Open-Closed Principle** — Core code changes for each new model
2. **Tight Coupling** — Factory knows about all implementations
3. **Extension Difficulty** — Third-party models require forking
4. **Maintenance Burden** — Growing if-else chain

---

## Acceptance Criteria

- [ ] Implement `@ModelLM.register("model_name")` decorator pattern
- [ ] Migrate existing models (GPT2, BERT, Minicons) to use registry
- [ ] Support automatic discovery via entry points (optional)
- [ ] Maintain backward compatibility with existing API
- [ ] Add documentation for extending with custom models

---

## Suggested Implementation

```python
# model_lm.py

class ModelLM:
    _registry: Dict[str, Type['ModelLM']] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_cls: Type['ModelLM']):
            cls._registry[name] = model_cls
            return model_cls
        return decorator
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> 'ModelLM':
        """Factory method using registry lookup."""
        if model_type not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {available}"
            )
        return cls._registry[model_type](**kwargs)


# gpt2_lm.py
@ModelLM.register("gpt2")
class GPT2LM(ModelLM):
    ...

# bert_lm.py
@ModelLM.register("bert")
class BertLM(ModelLM):
    ...

# User extension (no core modification needed)
@ModelLM.register("custom-llm")
class CustomLLM(ModelLM):
    ...
```

---

## Impact

| Area | Benefit |
|------|---------|
| Extensibility | New models without core changes |
| Maintainability | Decoupled model implementations |
| Third-Party Support | Easy custom model integration |
| Testing | Models can be mocked/stubbed easily |

---

## Labels
`architecture`, `high-priority`, `extensibility`, `design-pattern`
