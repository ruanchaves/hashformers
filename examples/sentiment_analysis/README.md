# Sentiment Analysis

A sentiment analysis model can benefit from hashtag segmentation as a preprocessing step.

Before:

> Who are you tomorrow? Will you make me smile or just
bring me sorrow? **#HottieOfTheWeek** Demi Lovato

After:

> Who are you tomorrow? Will you make me smile or just
bring me sorrow? **Hottie Of The Week** Demi Lovato

This transformation will improve results especially for models that have not been fine-tuned exclusively on tweets, or that have not been exposed to any tweets at all.

## Usage 

The command below will perform sentiment analysis on the [SemEval 2017 Task 4 English dataset](https://alt.qcri.org/semeval2017/task4/) with the `siebert/sentiment-roberta-large-english` model [\[ model card \]](https://huggingface.co/siebert/sentiment-roberta-large-english).

`gpt2-large` and `bert-large-uncased-whole-word-masking` will be utilized to segment the hashtags before sentiment analysis.

As `siebert/sentiment-roberta-large-english` has been fine-tuned to handle only two categories ( Positive and Negative ), Neutral samples will be ignored during evaluation.

```
python sentiment_analysis.py \
    --dataset_reader ./semeval2017.py \
    --dataset_url http://alt.qcri.org/semeval2017/task4/data/uploads/semeval2017-task4-test.zip \
    --sentiment_model siebert/sentiment-roberta-large-english \
    --segmenter_model_name_or_path gpt2-large \
    --reranker_model_name_or_path bert-large-uncased-whole-word-masking \
    --spacy_model en_core_web_sm
```