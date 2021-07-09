#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..

python calculate_scores.py \
    --model_name_or_path "bert-large-uncased-whole-word-masking" \
    --model_type "bert" \
    --source "./hashtags/Dev-BOUN/gpt2_large_dev_boun_13_steps/dict.json" \
    --output "./hashtags/Dev-BOUN/bert_from_gpt2_large_dev_boun_13_steps/dict.json"

python calculate_scores.py \
    --model_name_or_path "bert-large-uncased-whole-word-masking" \
    --model_type "bert" \
    --source "./hashtags/Test-BOUN/gpt2_large_test_boun_13_steps/dict.json" \
    --output "./hashtags/Test-BOUN/bert_from_gpt2_large_test_boun_13_steps/dict.json" \

python calculate_scores.py \
    --model_name_or_path "bert-large-uncased-whole-word-masking" \
    --model_type "bert" \
    --source "./hashtags/Dev-Stanford/gpt2_large_dev_stanford_13_steps/dict.json" \
    --output "./hashtags/Dev-Stanford/bert_from_gpt2_large_dev_stanford_13_steps/dict.json" \

    python calculate_scores.py \
    --model_name_or_path "bert-large-uncased-whole-word-masking" \
    --model_type "bert" \
    --source "./hashtags/Test-Stanford/gpt2_large_test_stanford_13_steps/dict.json" \
    --output "./hashtags/Test-Stanford/bert_from_gpt2_large_test_stanford_13_steps/dict.json" \