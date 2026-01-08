# Changelog

## v2.2.0 (2026-01-08)

### ✨ New Features

#### spaCy Pipeline Integration

Hashformers can now be used directly as a spaCy pipeline component! This enables seamless integration with spaCy workflows and makes it easy to combine word segmentation with other NLP tasks.

**Installation:**
```bash
pip install hashformers[spacy]
```

**Usage:**
```python
import spacy
import hashformers.spacy  # registers the 'hashformers' component

nlp = spacy.blank("en")
nlp.add_pipe("hashformers", config={"model": "distilgpt2"})

doc = nlp("#weneedanationalpark")
print(doc._.segmented)  # 'we need a national park'
```

**Configuration options:**
- `model`: Hugging Face model name (default: `"distilgpt2"`)
- `device`: Device to run on, `"cuda"` or `"cpu"` (default: `"cuda"`)
- `gpu_batch_size`: Batch size for GPU processing (default: `1000`)

### 📦 Package Improvements

- Added `long_description` from README for better PyPI presentation
- Added PyPI classifiers for improved discoverability
- Added `python_requires=">=3.8"` specification
- Added `keywords` for search optimization
- Added optional `[spacy]` extras for spaCy dependency

### 📚 Documentation

- Added spaCy integration section to README
- Prepared spaCy Universe submission

---

## v2.1.0

- Previous stable release
- Core word segmentation with TransformerWordSegmenter
- Beam search algorithm with optional reranker
- Support for any Hugging Face autoregressive model
