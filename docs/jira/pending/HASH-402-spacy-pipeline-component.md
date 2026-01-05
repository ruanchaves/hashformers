# HASH-402: Create spaCy Pipeline Component

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Type**    | Story                                                     |
| **Priority**| Medium                                                    |
| **Epic**    | Ecosystem Integrations                                    |

## Goal

Allow spaCy users to add hashformers as a pipeline component via `nlp.add_pipe("hashformers")`.

## Description

Create a spaCy pipeline component that integrates hashformers word segmentation into spaCy's NLP pipeline. This enables users to preprocess concatenated text (hashtags, URLs, identifiers) before or during spaCy's tokenization.

## Tasks

1. Use `@Language.factory` decorator to register the component.
2. Implement the `__call__` method to modify the `Doc` object or raw text before tokenization.
3. Handle component configuration (model, device, etc.).
4. Register the component for automatic discovery.
5. Write unit tests.
6. Document usage and configuration options.

## Example Usage

```python
import spacy
from hashformers.integrations import spacy_component  # registers the component

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("hashformers", first=True, config={
    "segmenter_model": "gpt2",
    "segmenter_type": "incremental"
})

doc = nlp("Check out #MakeAmericaGreatAgain")
print(doc.text)  # "Check out Make America Great Again"
```

## Technical Considerations

- Decide whether to modify raw text before tokenization or post-process tokens
- Handle the interaction with spaCy's existing tokenizer
- Consider memory efficiency for batch processing
- Support both CPU and GPU execution

## Deliverable

- A `spacy_component.py` module in `src/hashformers/integrations/`

## Acceptance Criteria

- [ ] Component registers via `@Language.factory("hashformers")`
- [ ] `__call__` method correctly processes `Doc` objects
- [ ] Works with `nlp.add_pipe("hashformers")`
- [ ] Configuration options are properly handled
- [ ] Unit tests pass
- [ ] Documentation includes setup and usage examples
