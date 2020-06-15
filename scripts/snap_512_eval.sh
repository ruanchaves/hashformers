#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..

## Test-BOUN

# GPT2 extra-large : Test-BOUN

python beamsearch.py \
    --model_name_or_path='/home/models/gpt2-512-small-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/fine_tuning/512_small_test_boun/expansions.json' \
    --dict_file='output/fine_tuning/512_small_test_boun/dict.json' \
    --report_file='output/fine_tuning/512_small_test_boun/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --logfile='512_small_test_boun.log'

## Stanford

# GPT2 extra-large : Stanford

python beamsearch.py \
    --model_name_or_path='/home/models/gpt2-512-small-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/fine_tuning/512_small_test_stanford/expansions.json' \
    --dict_file='output/fine_tuning/512_small_test_stanford/dict.json' \
    --report_file='output/fine_tuning/512_small_test_stanford/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --logfile='512_small_stanford.log'
