<h1 align="center">
  <img src="https://raw.githubusercontent.com/ruanchaves/hashformers/master/hashformers.png" width="300" title="hashformers">
</h1>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb) 
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/pip/)  

Hashtag segmentation is the task of automatically adding spaces between the words in a hashtag. 

Hashformers is the current **state-of-the-art** for hashtag segmentation. It is a **language-agnostic** library: you can use it to segment hashtags not just in English but also in any language with a GPT-2 model on the [Hugging Face Model Hub](https://huggingface.co/models).

* [**Segment hashtags on Google Colab**](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb)

* [**Read the step-by-step tutorial**](https://github.com/ruanchaves/hashformers/blob/master/hashformers.ipynb)

## Comparison to other hashtag segmentation libraries 

<h1 align="center">
  <img src="https://raw.githubusercontent.com/ruanchaves/hashformers/master/barplot_evaluation.png" width="512" title="hashformers">
</h1>

In this figure we compare **hashformers** with [HashtagMaster](https://github.com/mounicam/hashtag_master) ( also known as "MPNR" ) and [ekphrasis](https://github.com/cbaziotis/ekphrasis) on five hashtag segmentation benchmarks.

On average, hashformers is **10% more accurate** than the second best option. More details [on our wiki](https://github.com/ruanchaves/hashformers/wiki/Evaluation).

# Basic usage

```python
from hashformers import WordSegmenter

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

# Installation

```
pip install hashformers
```

It is possible to use **hashformers** without a reranker:

```python
ws = WordSegmenter(
    segmenter_model_name_or_path="gpt2",
    reranker_model_name_or_path=None
)
```

If you want to use a reranker model, you must install [mxnet](https://pypi.org/project/mxnet/). Here we install **hashformers** with `mxnet-cu110`, which is compatible with Google Colab. If installing in another environment, replace it by the [mxnet package](https://pypi.org/project/mxnet/) compatible with your CUDA version.

```
pip install mxnet-cu110 
pip install hashformers
```

# Development

Install **hashformers** directly from this repository ( or your fork ):

```
pip install git+git://github.com/ruanchaves/hashformers.git@master#egg=hashformers 
```

# Contributing 

Pull requests are welcome!  [Read our paper](https://arxiv.org/abs/2112.03213) for more details on the inner workings of our framework.

## Relevant Papers 

* [Zero-shot hashtag segmentation for multilingual sentiment analysis](https://arxiv.org/abs/2112.03213)

* [HashSet -- A Dataset For Hashtag Segmentation](https://arxiv.org/abs/2201.06741)

# Citation

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