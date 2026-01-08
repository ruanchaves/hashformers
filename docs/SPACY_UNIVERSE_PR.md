# Add Hashformers to spaCy Universe

## Description

This PR adds [Hashformers](https://github.com/ruanchaves/hashformers) to the spaCy Universe.

**Hashformers** is a word segmentation library that uses transformers and beam search to segment text without spaces (like hashtags) into words. It fills the gap between heuristic-based splitters and LLM prompt-based segmentation, and works with any model from the Hugging Face Model Hub.

## Key Features

- 🔤 **Word Segmentation**: Segments hashtags and concatenated text into individual words
- 🤗 **Hugging Face Compatible**: Works with any autoregressive model (GPT-2, LLaMA, etc.)
- 🌍 **Multilingual**: Supports any language with a compatible language model
- 🔬 **State-of-the-art**: Recognized as SOTA for hashtag segmentation at [LREC 2022](https://aclanthology.org/2022.lrec-1.782.pdf)
- ⚡ **spaCy Integration**: Available as a pipeline component via `pip install hashformers[spacy]`

## Example Usage

```python
from hashformers import TransformerWordSegmenter as WordSegmenter

ws = WordSegmenter(segmenter_model_name_or_path="distilgpt2")
result = ws.segment(["#weneedanationalpark"])
print(result)  # ['we need a national park']
```

## spaCy Pipeline Component

```python
import spacy

nlp = spacy.blank("en")
nlp.add_pipe("hashformers", config={"segmenter_model_name_or_path": "distilgpt2"})

doc = nlp("#weneedanationalpark")
print(doc[0]._.segmented)  # 'we need a national park'
```

## Checklist

- [x] Open-source license (MIT)
- [x] README with usage instructions
- [x] Available on PyPI (`pip install hashformers`)
- [x] GitHub repository
- [x] Working demo (Google Colab notebook)
- [x] spaCy pipeline component integration

## Links

- **GitHub**: https://github.com/ruanchaves/hashformers
- **PyPI**: https://pypi.org/project/hashformers/
- **Paper**: https://arxiv.org/abs/2112.03213
- **Colab Demo**: https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb

## Citations

The library has been cited in several academic papers including work on multilingual sentiment analysis, abusive language detection, and text processing.

```bibtex
@misc{rodrigues2021zeroshot,
      title={Zero-shot hashtag segmentation for multilingual sentiment analysis}, 
      author={Ruan Chaves Rodrigues and Marcelo Akira Inuzuka and others},
      year={2021},
      eprint={2112.03213},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
