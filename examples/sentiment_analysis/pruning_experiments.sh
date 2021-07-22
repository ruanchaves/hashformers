python sentiment_analysis.py \
    --log_level INFO \
    --dataset_reader semeval2017.py \
    --hashtag_only \
    --dataset_save_path semeval2017 \
    --run_classifier False \
    --do_eval False \
    --decoder_model_name_or_path gpt2-large \
    --decoder_model_type gpt2 \
    --encoder_model_name_or_path bert-large-uncased-whole-word-masking \
    --encoder_model_type bert \
    --spacy_model en_core_web_sm

for i in {1..24}
do
    python sentiment_analysis.py \
    --log_level INFO \
    --dataset_load_path semeval2017 \
    --run_segmenter False \
    --sentiment_model siebert/sentiment-roberta-large-english \
    --do_eval \
    --prune_layers $i
done