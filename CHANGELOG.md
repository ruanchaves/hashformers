# Changelog

## v3.0.0 (Unreleased)

### Added

- Added bounded local file sampling plus deferred, one-time discovery and
  exact-revision configuration of language-appropriate public Hugging Face
  models in the MCP server.
- Added opt-in adaptive CUDA microbatching with independent segmenter and
  reranker controllers, throughput/memory tuning, OOM backoff, and telemetry.
- Added a pinned Qwen3-0.6B benchmark protocol, fixed auditable samples, strict
  insertion-only validation, confidence intervals, and reproducibility
  metadata while retaining Qwen2 as a historical comparison. Published the
  complete protocol-v3 FP16 Tesla T4 run with raw generations and paired
  statistics. Protocol v3 records benign quote envelopes separately from the
  strict insertion-only content check, uses an explicit zero-shot input format
  that does not encourage quoted echoes, and independently reports strict
  validity, deterministic boundary-projection recovery, source fallback, and
  scored proposal accuracy.
- Added a fixed-manifest Hashformers benchmark runner with pinned GPT-2,
  DistilGPT2, and RuGPT3Small snapshots, raw per-sample predictions, paired
  cross-protocol statistics, and reproducibility metadata. Published Tesla T4
  runs use the PR #80 adaptive candidate controller with
  `gpu_batch_size="auto"`, a 512 maximum, and recorded controller telemetry.

### Breaking changes

- Removed the legacy `RegexWordSegmenter`, `TweetSegmenter`, and
  `TwitterTextMatcher` APIs together with their tweet-only result containers.
  Regex callers should use Python's `re` module or a dedicated heuristic
  splitter. Applications processing complete posts should extract hashtags,
  pass them to `TransformerWordSegmenter`, and replace them in application
  code.
- Removed the corresponding regex and tweet MCP tools. The MCP server remains
  focused on Transformer segmentation, file workflows, model discovery, and
  candidate ranking.
- Removed the mandatory `twitter-text-python` dependency and the MCP-only
  `regex` dependency after their final consumers were deleted.

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
