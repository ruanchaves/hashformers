<h1 align="center">
  <b>hashformers</b>
</h1>

**Hashtag segmentation** is the task of automatically inserting the missing spaces between the words in a hashtag. 

This package applies Transformer models to hashtag segmentation. It is built on top of the [transformers](https://github.com/huggingface/transformers) library and the [lm-scorer](https://github.com/simonepri/lm-scorer) and [mlm-scoring](https://github.com/awslabs/mlm-scoring) packages.

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

# Installation

Run our library on Google Colab or, alternatively, deploy it from a Docker container. 
You can customize the `Dockerfile` on this folder to your particular use case.

```
git clone https://github.com/ruanchaves/hashformers.git
cd hashformers
docker build .
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
