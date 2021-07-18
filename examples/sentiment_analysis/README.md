
```
python sentiment_analysis.py \
    --dataset_reader ./semeval2017.py
    --sentiment_model siebert/sentiment-roberta-large-english \
    --decoder_model_name_or_path gpt2-large \
    --encoder_model_name_or_path bert-large-uncased-whole-word-masking \
    --spacy_model en_core_web_sm
```