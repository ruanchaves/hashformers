# ✂️ hashformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb) 
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/pip/)  

Hashtag segmentation is the task of automatically adding spaces between the words on a hashtag. 

[Hashformers](https://github.com/ruanchaves/hashformers) is the current **state-of-the-art** for hashtag segmentation. On average, hashformers is **10% more accurate** than the second best hashtag segmentation library ( more details [on the docs](https://ruanchaves.github.io/hashformers/html/EVALUATION.html) ).

Hashformers is also **language-agnostic**: you can use it to segment hashtags not just in English, but also in any language with a GPT-2 model on the [Hugging Face Model Hub](https://huggingface.co/models).

<p align="center">

<h3> <a href="https://ruanchaves.github.io/hashformers/html/README.html"> ✂️ Read the documentation </a> </h3>

<h3> <a href="https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb"> ✂️ Segment hashtags on Google Colab </a> </h3>

<h3> <a href="https://github.com/ruanchaves/hashformers/blob/master/hashformers.ipynb"> ✂️ Follow the step-by-step tutorial </a> </h3>
</p>



## Basic usage

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

For more information, read the [documentation for the WordSegmenter object](https://ruanchaves.github.io/hashformers/html/hashformers.segmenter.html#hashformers.segmenter.segmenter.WordSegmenter).

## Installation

Hashformers is compatible with Python 3.7.

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

## Contributing 

Pull requests are welcome!  [Read our paper](https://arxiv.org/abs/2112.03213) for more details on the inner workings of our framework.

If you want to develop the library, you can install **hashformers** directly from this repository ( or your fork ):

```
git clone https://github.com/ruanchaves/hashformers.git
cd hashformers
pip install -e .
```

## Relevant Papers 

* [Zero-shot hashtag segmentation for multilingual sentiment analysis](https://arxiv.org/abs/2112.03213)

* [HashSet -- A Dataset For Hashtag Segmentation](https://arxiv.org/abs/2201.06741)

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
```s