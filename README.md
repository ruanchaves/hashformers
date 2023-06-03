# ✂️ hashformers

[![HF Spaces](https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg)](https://ruanchaves-hashtag-segmentation.hf.space/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb) [![PyPi license](https://badgen.net/pypi/license/pip/)](https://github.com/ruanchaves/hashformers/blob/master/LICENSE) [![stars](https://img.shields.io/github/stars/ruanchaves/hashformers)](https://github.com/ruanchaves/hashformers) [![tweet](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fruanchaves%2Fhashformers)](https://www.twitter.com/share?url=https://github.com/ruanchaves/hashformers)


Hashtag segmentation is the task of automatically adding spaces between the words on a hashtag. 

[Hashformers](https://github.com/ruanchaves/hashformers) is the current **state-of-the-art** for hashtag segmentation, as demonstrated on [this paper accepted at LREC 2022](https://aclanthology.org/2022.lrec-1.782.pdf). 

Hashformers is also **language-agnostic**: you can use it to segment hashtags not just with English models, but also using any language model available on the [Hugging Face Model Hub](https://huggingface.co/models).

<p align="center">
    
<h3> <a href="https://ruanchaves-hashtag-segmentation.hf.space/"> ✂️ Segment hashtags on Hugging Face Spaces </a> </h3>

<h3> <a href="https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb"> ✂️ Get started - Google Colab tutorial </a> </h3>

</p>



## Basic usage

```python
from hashformers import TransformerWordSegmenter as WordSegmenter

ws = WordSegmenter(
    segmenter_model_name_or_path="gpt2",
    segmenter_model_type="incremental",
    reranker_model_name_or_path="google/flan-t5-base",
    reranker_model_type="seq2seq"
)

segmentations = ws.segment([
    "#weneedanationalpark",
    "#icecold"
])

print(segmentations)

# [ 'we need a national park',
# 'ice cold' ]
```

It is also possible to use hashformers without a reranker by setting the `reranker_model_name_or_path` and the `reranker_model_type` to `None`. 

## Installation

```
pip install hashformers
```

## Contributing 

Pull requests are welcome!  [Read our paper](https://arxiv.org/abs/2112.03213) for more details on the inner workings of our framework.

If you want to develop the library, you can install **hashformers** directly from this repository ( or your fork ):

```
git clone https://github.com/ruanchaves/hashformers.git
cd hashformers
pip install -e .
```

## Relevant Papers 

This is a collection of papers that have utilized the *hashformers* library as a tool in their research.

### hashformers v1.3

These papers have utilized `hashformers` version 1.3 or below.

* [Zero-shot hashtag segmentation for multilingual sentiment analysis](https://arxiv.org/abs/2112.03213)

* [HashSet -- A Dataset For Hashtag Segmentation (LREC 2022)](https://aclanthology.org/2022.lrec-1.782/)

* [Generalizability of Abusive Language Detection Models on Homogeneous German Datasets](https://link.springer.com/article/10.1007/s13222-023-00438-1#Fn3) 

* [The problem of varying annotations to identify abusive language in social media content](https://www.cambridge.org/core/journals/natural-language-engineering/article/problem-of-varying-annotations-to-identify-abusive-language-in-social-media-content/B47FCCCEBF6EDF9C628DCC69EC5E0826)

## Blog Posts

* [15 Datasets for Word Segmentation on the Hugging Face Hub](https://ruanchaves.medium.com/15-datasets-for-word-segmentation-on-the-hugging-face-hub-4f24cb971e48)

## Citation

```
@misc{rodrigues2021zeroshot,
      title={Zero-shot hashtag segmentation for multilingual sentiment analysis}, 
      author={Ruan Chaves Rodrigues and Marcelo Akira Inuzuka and Juliana Resplande Sant'Anna Gomes and Acquila Santos Rocha and Iacer Calixto and Hugo Alexandre Dantas do Nascimento},
      year={2021},
      eprint={2112.03213},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```