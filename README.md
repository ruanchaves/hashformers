<h1 align="center">
  <img src="https://raw.githubusercontent.com/ruanchaves/hashformers/master/hashformers.png" width="300" title="hashformers">
</h1>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb) 
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/pip/)  

**Hashtag segmentation** is the task of automatically inserting the missing spaces between the words in a hashtag. 

**Hashformers** applies Transformer models to hashtag segmentation. It is built on top of the [transformers](https://github.com/huggingface/transformers) library and the [lm-scorer](https://github.com/simonepri/lm-scorer) and [mlm-scoring](https://github.com/awslabs/mlm-scoring) packages.

Try it right now on [Google Colab](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb).

**Paper:** [Zero-shot hashtag segmentation for multilingual sentiment analysis](https://arxiv.org/abs/2112.03213)

# Basic usage

```python
from hashformers import WordSegmenter

ws = WordSegmenter(
    segmenter_model_name_or_path="gpt2",
    reranker_model_name_or_path="bert-base-uncased",
    use_reranker=True
)

segmentations = ws.segment([
    "#myoldphonesucks",
    "#latinosinthedeepsouth",
    "#weneedanationalpark",
    "#LandoftheLost",
    "#icecold",
    "#Heartbreaker",
    "#TheRiseGuys"
])

print(segmentations)

# ['my old phone sucks',
# 'latinos in the deep south',
# 'we need a national park',
# 'Land of the Lost',
# 'ice cold',
# 'Heartbreaker',
# 'The Rise Guys']
```

# Installation

Installation steps are described on this [notebook](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb). 
A Docker image is coming soon.

# Examples

Applications of hashtag segmentation to tweet sentiment analysis and the automatic translation of tweets can be found on the `examples` folder.

<!--- # Citation ---> 

<!--- You can cite [this paper](#PAPER_URL#) when referring to the hashformers library: ---> 

<!--- ```
@misc{rodrigues2022segmentation,
      title={dolorem ipsum quia dolor sit amet}, 
      author={Ruan Chaves Rodrigues},
      year={2022},
      eprint={9999.99999},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
``` ---> 

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
