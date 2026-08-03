# ✂️ hashformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb) [![PyPi license](https://badgen.net/pypi/license/pip/)](https://github.com/ruanchaves/hashformers/blob/master/LICENSE) [![stars](https://img.shields.io/github/stars/ruanchaves/hashformers)](https://github.com/ruanchaves/hashformers)

**Hashformers** is a word segmentation library that fills a gap in the NLP ecosystem between heuristic-based splitters and LLM prompt-based segmentation. It can be used with any language model from the [Hugging Face Model Hub](https://huggingface.co/models), from auto-regressive models like GPT-2 to recent large language models (LLMs).

**Hashformers** uses language models and a beam search algorithm to segment text without spaces into words. Historical benchmarks compare specific Hashformers, heuristic, and prompted-model configurations; their conclusions are scoped to the evaluated samples and settings.

<p align="center">
<h3> <a href="https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb"> ✂️ Google Colab Tutorial </a> </h3>
</p>

<p align="center">
<h3> <a href="https://github.com/ruanchaves/hashformers/blob/master/tutorials/EVALUATION-January_2026.md"> ✂️ Evaluation Report </a> </h3>
</p>

---

## 🚀 Quick Start

### Installation

```bash
pip install hashformers
```

Hashformers requires Python 3.10 or newer and supports Transformers 4.46.1
through 5.x.

### Basic Usage

```python
from hashformers import TransformerWordSegmenter as WordSegmenter

ws = WordSegmenter(
    segmenter_model_name_or_path="distilgpt2"
) # You can use any model from the Hugging Face Model Hub

segmentations = ws.segment([
    "#weneedanationalpark",
    "#icecold"
])

print(segmentations)
# ['we need a national park', 'ice cold']
```

The default search uses `topk=5` and `steps=5` for interactive performance.
For a wider search, use `ws.segment(inputs, topk=20, steps=13)`. Inference
batches default to 64 candidates and can be configured when constructing the
segmenter.

For bulk CUDA workloads, opt into adaptive scorer microbatching independently
for beam search and reranking:

```python
ws = WordSegmenter(
    segmenter_model_name_or_path="distilgpt2",
    segmenter_gpu_batch_size="auto",
    segmenter_max_gpu_batch_size=512,
    reranker_model_name_or_path="bert-base-uncased",
    reranker_gpu_batch_size="auto",
    reranker_max_gpu_batch_size=512,
)
```

Automatic mode starts at 64, tries geometrically larger full microbatches, and
keeps a larger size only when synchronized candidate throughput improves while
at least 20% of CUDA memory remains free. It backs off and retries the same
candidates after a CUDA out-of-memory error. The selected size is cached only
for that scorer and process; the segmenter and reranker therefore tune
independently. Small calls do not trigger growth, and CPU execution uses the
safe starting size without CUDA tuning. Inspect
`ws.get_segmenter().batch_telemetry` (or `ws.get_reranker().batch_telemetry`)
for the configured and effective sizes, tuning state, throughput, memory, and
OOM backoff count.

This controller targets bulk candidate throughput, not a particular reported
GPU-utilization percentage. CUDA utilization is sampled too coarsely to be the
control signal for short beam-search scorer calls. Keep an explicit integer
batch size, which remains the default, for predictable shared-GPU memory or
latency.

### Using Language-Specific Models

```python
# Russian hashtags with RuGPT3
ws = WordSegmenter(
    segmenter_model_name_or_path="ai-forever/rugpt3small_based_on_gpt2"
)

segmentations = ws.segment(["#москвасити"])

print(segmentations)
# ['москва сити']
```

### spaCy Integration

Hashformers can be used as a spaCy pipeline component:

```python
import spacy
import hashformers.spacy  # registers the "hashformers" component

nlp = spacy.blank("en")
nlp.add_pipe("hashformers", config={"model": "distilgpt2"})

doc = nlp("#weneedanationalpark")
print(doc._.segmented)  # "we need a national park"
```

Install with spaCy support:

```bash
pip install hashformers[spacy]
```

### MCP and Agent Skill

Install the optional local MCP server on Python 3.10 or newer:

```bash
pip install "hashformers[mcp]"
```

The core `pip install hashformers` installation does not include the MCP SDK.
Start the server over stdio with the same model settings accepted by the Python
API:

```bash
hashformers-mcp \
  --model distilgpt2 \
  --batch-size auto \
  --max-batch-size 512 \
  --file-root /path/to/project
```

When the language is not known until an agent sees the request, start the same
server in explicitly authorized deferred-selection mode:

```bash
hashformers-mcp \
  --defer-model-selection \
  --file-root /path/to/project
```

That process starts without selecting, downloading, or loading a Transformer.
The agent can inspect at most 20 local examples with `sample_hashtag_file`,
infer and report a language and confidence, obtain a bounded public-model
shortlist with `discover_huggingface_models`, and make one `configure_models`
call. Configuration anonymously re-fetches both repositories, rejects private,
gated, over-size, unsupported, or custom-code models, and requires the exact
Hub commit SHA returned by discovery. The first later inference downloads that
pinned revision lazily. Identical configuration retries are safe; selecting a
different model requires restarting the server.

The default Hub ceilings are one billion parameters and 5 GB of selected pinned
snapshot files. Operators can lower them with `--max-model-parameters` and
`--max-model-size-bytes`. Deferred selection is the operator's authorization
for its single remote selection and download; ordinary startup does not allow
callers to change models.

Optional startup flags configure the segmenter model and scorer type, device,
and batch size. Supplying `--reranker-model` enables reranker-only and ensemble
selection, with corresponding model-type, device, and batch-size flags. Run
`hashformers-mcp --help` for the complete list. Models are loaded lazily and
reused for the life of the process; `--device auto` selects CUDA when available
and otherwise uses CPU. `--batch-size auto` and
`--reranker-batch-size auto` opt the two scorers into independent adaptive
microbatching; bound them with `--max-batch-size` and
`--reranker-max-batch-size`. File-job tools are restricted to the server
working directory by default; repeat `--file-root` to authorize other
directories.
File jobs currently require Linux descriptor-backed filesystem support through
`/proc/self/fd`; interactive segmentation, model discovery/configuration, and
candidate ranking remain available on other platforms.

MCP clients normally launch that command for you. For example, configure Codex
or Claude Code with:

```bash
codex mcp add hashformers -- hashformers-mcp --model distilgpt2
claude mcp add --transport stdio --scope user hashformers -- \
  hashformers-mcp --model distilgpt2
```

The server exposes the complete user-facing segmentation workflow:

| Tool | Purpose |
|------|---------|
| `sample_hashtag_file` | Return at most 20 distinct reservoir-sampled hashtags and compact text, CSV, or JSON Lines metadata without copying the dataset into agent context. |
| `discover_huggingface_models` | Return a deterministic, hard-capped shortlist of anonymously validated public Transformers models for a language and role. |
| `configure_models` | Validate and publish one segmenter and optional reranker at exact Hub revisions when deferred selection was authorized at startup. |
| `segment_hashtags` | Run Transformer beam search with configurable search depth, preprocessing, reranker or ensemble selection, and component rankings. |
| `start_hashtag_file_job` | Index and deduplicate text, CSV, or JSON Lines locally without loading a model or placing the dataset in agent context. |
| `continue_hashtag_file_job` | Process and atomically checkpoint one bounded batch of a file job; repeat until completion. |
| `rank_candidates` | Select, rerank, or ensemble precomputed hypotheses without rerunning beam search. |

Hashformers intentionally operates on hashtags rather than complete social
posts. Applications that process full text should extract hashtags, call
`segment_hashtags`, and perform replacements in application code. Generic
regex substitution belongs in the caller or a dedicated heuristic splitter.

`top_k` controls beam width while `max_candidates` independently limits the
returned or written alternatives and is capped at 64. Model identity, devices,
and GPU batch sizes are startup settings, except for the single exact-revision
selection explicitly authorized by `--defer-model-selection`. The process
retains at most one segmenter and one reranker. Interactive
Transformer calls accept at most 64 inputs, `top_k` is capped at 64, and `steps`
at 32. An aggregate expansion budget also rejects combinations of long inputs,
wide beams, and deep searches that would still consume excessive memory. Larger
inputs belong in the resumable file workflow.

For large local datasets, pass only paths to `start_hashtag_file_job`. A text
file contains one hashtag per line; CSV and JSON Lines inputs use the `hashtag`
field by default. Repeatedly call `continue_hashtag_file_job` with the returned
job path until its status is `completed`. Each call processes at most 64 unique
hashtags by default and persists an atomic SQLite checkpoint, so client timeouts
or restarts do not lose completed work. The final JSON Lines file preserves
every source row and duplicate in order. Full inputs and results never consume
agent-context tokens unless the user later asks the agent to open the output.
The checkpoint contains the indexed inputs, so continuation does not repeatedly
reread the source file. Existing outputs are never replaced unless the server
was started with `--allow-file-overwrite` and the caller also passes
`overwrite=true`.

If the file language is unknown, call `sample_hashtag_file` before starting the
job. Its deterministic hash-reservoir keeps memory and response size bounded
independently of file length and returns only distinct samples, record count,
format, and file size. No file contents or hashtags are sent to Hugging Face.
File-job checkpoints, progress responses, interactive results, and final JSON
Lines records preserve selected repository IDs and revisions. Deferred/pinned
selection always records an exact revision; ordinary unpinned startup records
`null`.

This repository also includes the `segment-hashtags` Agent Skill at
`.agents/skills/segment-hashtags`. Codex discovers it automatically when run
from this repository. To make it available from every project, copy it to the
user skill directory:

```bash
mkdir -p ~/.agents/skills
cp -R .agents/skills/segment-hashtags ~/.agents/skills/
```

Claude Code uses a different discovery directory:

```bash
mkdir -p ~/.claude/skills
cp -R .agents/skills/segment-hashtags ~/.claude/skills/
```

Restart the client if a newly created skill directory is not detected. The
Skill chooses the appropriate MCP workflow and does not implement segmentation
itself, so the MCP server must also be installed and configured.

## When to Use Hashformers?

The table below describes the practical trade-offs between Hashformers,
vocabulary-based splitters, and direct prompted generation. The January 2026
Qwen2 result is a single historical configuration, not evidence about LLMs as a
class. A pinned, auditable [Qwen3 benchmark protocol](benchmarks/qwen/README.md)
is ready for a fresh run; no new result is reported yet.

| Approach | Examples | Recommended When... | Notes |
|----------|----------|---------------------|-------|
| **Heuristic-based** | [SymSpell](https://github.com/wolfgarbe/SymSpell), [Ekphrasis](https://github.com/cbaziotis/ekphrasis), [WordNinja](https://github.com/keredson/wordninja), [Spiral (Ronin)](https://github.com/casics/spiral) | • **Scalability** is a primary requirement.<br><br>• The segmentation domain works well with a standard pre-built vocabulary. | Fast and efficient, but requires a pre-built vocabulary which can be limiting for niche domains or languages. |
| **Hashformers** | [Hashformers](https://github.com/ruanchaves/hashformers) | • You want beam-search segmentation backed by a language model.<br><br>• You are working in a domain or language where an appropriate backbone is available, but compiling a manual vocabulary is too burdensome. | Accuracy and performance depend on the selected backbone, language, dataset, and search settings. |
| **Prompted generative segmentation** | [Qwen3 benchmark protocol](benchmarks/qwen/README.md) | • You want to test a generative model under an explicit insertion-only output contract.<br><br>• Generation latency and invalid outputs are acceptable and measured. | The published report evaluated only one five-shot Qwen2-0.5B configuration. It does not establish a general size or quality threshold for prompted models. |

---

## 📚 Research & Citations

Hashformers was recognized as **state-of-the-art** for hashtag segmentation at [LREC 2022](https://aclanthology.org/2022.lrec-1.782.pdf).

### Papers Using Hashformers

- [Zero-shot hashtag segmentation for multilingual sentiment analysis](https://arxiv.org/abs/2112.03213)

- [HashSet -- A Dataset For Hashtag Segmentation (LREC 2022)](https://aclanthology.org/2022.lrec-1.782/)

- [Generalizability of Abusive Language Detection Models on Homogeneous German Datasets](https://link.springer.com/article/10.1007/s13222-023-00438-1#Fn3) 

- [The problem of varying annotations to identify abusive language in social media content](https://www.cambridge.org/core/journals/natural-language-engineering/article/problem-of-varying-annotations-to-identify-abusive-language-in-social-media-content/B47FCCCEBF6EDF9C628DCC69EC5E0826)

- [NUSS: An R package for mixed N-grams and unigram sequence segmentation](https://www.sciencedirect.com/science/article/pii/S2352711025002754#bbib0017)

### Citation

If you find **Hashformers** useful, please consider citing our paper:

```bibtex
@misc{rodrigues2021zeroshot,
      title={Zero-shot hashtag segmentation for multilingual sentiment analysis}, 
      author={Ruan Chaves Rodrigues and Marcelo Akira Inuzuka and Juliana Resplande Sant'Anna Gomes and Acquila Santos Rocha and Iacer Calixto and Hugo Alexandre Dantas do Nascimento},
      year={2021},
      eprint={2112.03213},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---

## 🤝 Contributing

Pull requests are welcome! [Read our paper](https://arxiv.org/abs/2112.03213) for details on the framework architecture.

```bash
git clone https://github.com/ruanchaves/hashformers.git
cd hashformers
pip install -e .
```

---

## 📖 Resources

- [15 Datasets for Word Segmentation on the Hugging Face Hub](https://medium.com/@ruanchaves/15-datasets-for-word-segmentation-on-the-hugging-face-hub-4f24cb971e48)
- [Benchmark Scripts](scripts/)
- [Evaluation Report (January 2026)](tutorials/EVALUATION-January_2026.md)
- [Evaluation Report (February 2022)](tutorials/EVALUATION-February_2022.md)


