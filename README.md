# ✂️ hashformers

[![HF Spaces](https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg)](https://ruanchaves-hashtag-segmentation.hf.space/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb) [![PyPi license](https://badgen.net/pypi/license/pip/)](https://github.com/ruanchaves/hashformers/blob/master/LICENSE) [![stars](https://img.shields.io/github/stars/ruanchaves/hashformers)](https://github.com/ruanchaves/hashformers) [![tweet](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fruanchaves%2Fhashformers)](https://www.twitter.com/share?url=https://github.com/ruanchaves/hashformers)


Hashtag segmentation is the task of automatically adding spaces between the words on a hashtag. 

[Hashformers](https://github.com/ruanchaves/hashformers) is the current **state-of-the-art** for hashtag segmentation. On average, hashformers is **10% more accurate** than the second best hashtag segmentation library ( [Learn More](https://github.com/ruanchaves/hashformers/blob/master/tutorials/EVALUATION.md) ).

Hashformers is also **language-agnostic**: you can use it to segment hashtags not just in English, but also in any language with a GPT-2 model on the [Hugging Face Model Hub](https://huggingface.co/models).

<p align="center">
    
<h3> <a href="https://ruanchaves-hashtag-segmentation.hf.space/"> ✂️ Segment hashtags on Hugging Face Spaces </a> </h3>

<h3> <a href="https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb"> ✂️ Get started - Google Colab tutorial </a> </h3>

</p>



## Basic usage

```python
from hashformers import TransformerWordSegmenter as WordSegmenter

ws = WordSegmenter(
    segmenter_model_name_or_path="gpt2",
    reranker_model_name_or_path="bert-base-uncased"
)

segmentations = ws.segment([
    "#weneedanationalpark",
    "#icecold"
])

print(segmentations)

# [ 'we need a national park',
# 'ice cold' ]
```

## Installation

Hashformers is compatible with Python 3.7.

```
pip install hashformers
```

It is possible to use **hashformers** without a reranker:

```python
from hashformers import TransformerWordSegmenter as WordSegmenter
ws = WordSegmenter(
    segmenter_model_name_or_path="gpt2",
    reranker_model_name_or_path=None
)
```

If you want to use a BERT model as a reranker, you must install [mxnet](https://pypi.org/project/mxnet/). Here we install **hashformers** with `mxnet-cu110`, which is compatible with Google Colab. If installing in another environment, replace it by the [mxnet package](https://pypi.org/project/mxnet/) compatible with your CUDA version.

```
pip install mxnet-cu110 
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
