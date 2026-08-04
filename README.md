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

### MCP and Agent Skill

Install and start the optional local MCP server:

```bash
pip install "hashformers[mcp]"
hashformers-mcp \
  --model distilgpt2 \
  --batch-size auto \
  --file-root /path/to/project
```

Add it to Codex or Claude Code:

```bash
codex mcp add hashformers -- hashformers-mcp --model distilgpt2
claude mcp add --transport stdio --scope user hashformers -- \
  hashformers-mcp --model distilgpt2
```

Use `segment_hashtags` for interactive requests. For large text, CSV, or JSON
Lines datasets, give the agent a local path and use `start_hashtag_file_job`
followed by `continue_hashtag_file_job`; this keeps the dataset out of the
agent's context and provides resumable checkpoints. If the language is unknown,
start the server with `--defer-model-selection` so the agent can sample the file
and choose a suitable public Hugging Face model before segmentation.

The repository includes a `segment-hashtags` Agent Skill. Install it globally
for Codex or Claude Code with:

```bash
mkdir -p ~/.agents/skills ~/.claude/skills
cp -R .agents/skills/segment-hashtags ~/.agents/skills/
cp -R .agents/skills/segment-hashtags ~/.claude/skills/
```

Run `hashformers-mcp --help` for all model, reranker, device, and file-access
options.

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

## When to Use Hashformers?

Hashformers is designed to perform hashtag segmentation primarily on consumer
GPUs. Its Transformer models often outperform language models that can run
locally on GPUs at the same speed and scale, as shown in the
[Qwen benchmark](benchmarks/qwen/README.md). It can be especially useful when
both of the following are true:

- You have access to GPU compute. Even when renting a GPU, our
  [cost projections](benchmarks/qwen/results/2026-08-03-colab-t4-fp16-v3/hosted-api-cost-projection.svg)
  show Hashformers can be cheaper than major LLM providers at volumes of
  roughly 120 hashtags or more.
- Your hashtag segmentation domain is niche enough that heuristic methods such
  as [SymSpell](https://github.com/wolfgarbe/SymSpell),
  [Ekphrasis](https://github.com/cbaziotis/ekphrasis),
  [WordNinja](https://github.com/keredson/wordninja), or
  [Spiral (Ronin)](https://github.com/casics/spiral) are not accurate enough.

Conversely, you may not wish to use Hashformers if:

- Your domain is simple enough for a CPU-based heuristic method.
- You are segmenting a low volume of hashtags, which may not justify using a
  local GPU instead of a major LLM provider.
- You are targeting maximum accuracy at any cost, in which case a cutting-edge
  model from a major LLM provider may be a better fit.

Hashformers therefore sits in the middle ground between heuristic methods and
LLM APIs for users with consumer GPUs.

---

## 📚 Research & Citations

Hashformers was recognized as **state-of-the-art** for hashtag segmentation at [LREC 2022](https://aclanthology.org/2022.lrec-1.782.pdf).

### Papers Using Hashformers

- [Zero-shot hashtag segmentation for multilingual sentiment analysis](https://arxiv.org/abs/2112.03213)

- [HashSet -- A Dataset For Hashtag Segmentation (LREC 2022)](https://aclanthology.org/2022.lrec-1.782/)

- [Generalizability of Abusive Language Detection Models on Homogeneous German Datasets](https://link.springer.com/article/10.1007/s13222-023-00438-1#Fn3) 

- [The problem of varying annotations to identify abusive language in social media content](https://www.cambridge.org/core/journals/natural-language-engineering/article/problem-of-varying-annotations-to-identify-abusive-language-in-social-media-content/B47FCCCEBF6EDF9C628DCC69EC5E0826)

- [NUSS: An R package for mixed N-grams and unigram sequence segmentation](https://www.sciencedirect.com/science/article/pii/S2352711025002754#bbib0017)

- [Hate Speech Detection in Turkish and Arabic Languages: A Comprehensive Study](https://arxiv.org/html/2607.00143v1)

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


