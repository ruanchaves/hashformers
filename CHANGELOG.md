# Changelog

## v3.0.0 (2026-08-04)

Hashformers 3 is an agent-ready release centered on Transformer segmentation,
bounded local workflows, efficient inference, and reproducible evaluation.

### MCP and agent workflows

- Added the optional `hashformers[mcp]` installation and `hashformers-mcp`
  command for interactive segmentation, candidate ranking, and resumable local
  file jobs.
- Added bounded batches, payloads, beam expansion, file sampling, and inference
  chunks. File jobs preserve source order and duplicates, checkpoint before
  publication, publish atomically, and validate authorized paths against
  symlink and descriptor swaps.
- Added the `segment-hashtags` Agent Skill for Codex and Claude Code workflows,
  including bulk files that remain outside the agent context.
- Added deferred, one-time discovery of language-appropriate public Hugging
  Face models. Selected models are validated, pinned to exact Hub revisions,
  and loaded only when inference begins.
- Deferred the ML and Hub runtime imports required by the MCP server. Stdio
  initialization now completes before common client startup deadlines while
  the historical top-level Python imports remain available.

### Ranking and inference

- Added opt-in weighted reciprocal rank fusion over complete segmenter and
  reranker candidate lists. Existing callers continue to use `top2` fusion by
  default; RRF is available consistently through Python, MCP, candidate
  ranking, and resumable jobs.
- Added opt-in adaptive CUDA microbatching with independent segmenter and
  reranker controllers, throughput-based growth, telemetry, and exact-slice
  OOM recovery.
- Reduced default scorer batches from 1,000 to 64 and the default beam search
  from `topk=20, steps=13` to `topk=5, steps=5`. The previous search budget
  remains available through explicit options.
- Deduplicated equivalent candidates before scoring and beam pruning so
  duplicate hypotheses no longer consume scorer work or beam slots.

### Fixes and compatibility

- Fixed reranker score direction so masked and sequence-to-sequence language
  models follow Hashformers' lower-is-better cost convention.
- Honored CPU and explicit CUDA device selection in the masked reranker.
- Restored masked reranking on Transformers 5 while retaining compatibility
  with Transformers 4.46.1 and newer 4.x releases.
- Fixed precomputed segmenter-run handling and ensured reranker output is used
  when ensemble selection is disabled.
- Added explicit dependency ranges for Minicons and Transformers and extensive
  regression coverage for package imports, scoring, model selection, adaptive
  batching, RRF, and MCP workflows.

### Benchmarks and documentation

- Published a fixed 280-item multilingual benchmark with pinned model
  revisions, raw predictions, checksums, confidence intervals, and paired
  statistics. Hashformers-DistilGPT2 reached 65.00% exact match versus 27.50%
  for the measured Qwen3-0.6B prompt configuration, a 37.5 percentage-point
  advantage; measured DistilGPT2 throughput was 14.11 hashtags/second on a T4.
- Added reproducible Hashformers, Qwen, and hosted-API cost projection tooling
  with machine-readable assumptions. Hosted-provider accuracy and latency
  remain explicitly illustrative rather than measured results.
- Added a Colab tutorial for Python, Codex, Claude Code, MCP, resumable file
  jobs, and deferred model selection.
- Reworked the README around copyable workflows, decision guidance, benchmark
  evidence, research references, and an animated terminal demonstration.

### Breaking changes

- Python 3.10 or newer is now required. Python 3.8 and 3.9 users should remain
  on Hashformers 2.2.0.
- Removed `RegexWordSegmenter`, `TweetSegmenter`, and `TwitterTextMatcher`,
  together with the tweet-only result containers. Extract hashtags in
  application code, segment them with `TransformerWordSegmenter`, and replace
  them in application code when processing complete posts.
- Removed the corresponding regex and tweet MCP tools. The MCP server now
  focuses on Transformer segmentation, candidate ranking, model discovery,
  and file workflows.
- Removed the unused `twitter-text-python` and MCP-only `regex` dependencies.
- The smaller default beam and batch settings can change exact outputs or
  throughput relative to 2.2.0. Pass the former values explicitly when that
  behavior is required.

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
