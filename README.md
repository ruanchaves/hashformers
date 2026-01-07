# ✂️ hashformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb) [![PyPi license](https://badgen.net/pypi/license/pip/)](https://github.com/ruanchaves/hashformers/blob/master/LICENSE) [![stars](https://img.shields.io/github/stars/ruanchaves/hashformers)](https://github.com/ruanchaves/hashformers)

**Hashformers** is a word segmentation library that fills a gap in the NLP ecosystem between heuristic-based splitters and LLM prompt-based segmentation. It can be used with any language model from the [Hugging Face Model Hub](https://huggingface.co/models).

**Hashformers** uses transformers and a beam search approach to segment text without spaces into words. Benchmarks show that it can outperform heuristic-based splitters and LLMs on word segmentation tasks.

<p align="center">
<h3> <a href="https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb"> Try hashformers on Google Colab </a> </h3>
</p>

<p align="center">
<h3> <a href="https://github.com/ruanchaves/hashformers/blob/master/tutorials/EVALUATION-January_2026.md"> Read the Evaluation Report </a> </h3>
</p>

---

## 🚀 Quick Start

### Installation

```bash
pip install hashformers
```

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

## When to Use Hashformers?

The table below outlines when to use **Hashformers** versus other approaches like heuristic-based splitters (e.g., SymSpell, WordNinja) or large LLMs.

| Approach | Examples | Recommended When... | Notes |
|----------|----------|---------------------|-------|
| **Heuristic-based** | [SymSpell](https://github.com/wolfgarbe/SymSpell), [Ekphrasis](https://github.com/cbaziotis/ekphrasis), [WordNinja](https://github.com/keredson/wordninja), [Spiral (Ronin)](https://github.com/casics/spiral) | • **Scalability** is a primary requirement.<br><br>• The segmentation domain works well with a standard pre-built vocabulary. | Fast and efficient, but requires a pre-built vocabulary which can be limiting for niche domains or languages. |
| **Hashformers** | [Hashformers](https://github.com/ruanchaves/hashformers) | • **Scalability** is needed.<br><br>• You are working in a domain or language where a Language Model is readily available, but compiling a manual vocabulary for your task is too burdensome. | Evidence shows Hashformers can be superior to LLMs of similar scale (0.5B parameters). |
| **Large LLMs** | [OpenAI](https://openai.com/), Local LLM Deployment | • **Cost, latency, and scalability** are not concerns.<br><br>• You are segmenting a **low volume** of items. | To gain an accuracy advantage over Hashformers, you generally need to use significantly larger LLMs. |

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

