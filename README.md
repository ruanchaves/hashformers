<h1 align="center">
  <b>hashformers</b>
</h1>

**Hashtag segmentation** is the task of automatically inserting the missing spaces between the words in a hashtag. It can be used to improve tweet sentiment analysis and the automatic translation of tweets.

This package applies Transformer models to hashtag segmentation. It is built on top of the [transformers](https://github.com/huggingface/transformers) library and the [lm-scorer](https://github.com/simonepri/lm-scorer) and [mlm-scoring](https://github.com/awslabs/mlm-scoring) packages.

# Basic usage

```python
from hashformers import WordSegmenter

ws = WordSegmenter(
    segmenter_model_name_or_path='gpt2',
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



# Citation 
