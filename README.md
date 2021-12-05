<h1 align="center">
  ![](https://raw.githubusercontent.com/ruanchaves/hashformers/master/hashformers.png)
</h1>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb) 
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/pip/) 
[![GitHub stars](https://img.shields.io/github/stars/Naereen/StrapDown.js.svg?style=social&label=Star&maxAge=2592000)](https://github.com/ruanchaves/hashformers)
 

**Hashtag segmentation** is the task of automatically inserting the missing spaces between the words in a hashtag. 

This package applies Transformer models to hashtag segmentation. It is built on top of the [transformers](https://github.com/huggingface/transformers) library and the [lm-scorer](https://github.com/simonepri/lm-scorer) and [mlm-scoring](https://github.com/awslabs/mlm-scoring) packages.

Try it right now on [Google Colab](https://colab.research.google.com/github/ruanchaves/hashformers/blob/master/hashformers.ipynb).

**Paper coming soon**

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
