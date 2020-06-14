#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..
cd .. 

# GPT2 small : Test-BOUN

python translation_task.py \
    --model_name_or_path='/home/models/gpt2-translation-small-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/translation/fine_tuning_13/small_test_boun/expansions.json' \
    --dict_file='output/translation/fine_tuning_13/small_test_boun/dict.json' \
    --report_file='output/translation/fine_tuning_13/small_test_boun/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='translation_small_test_boun.log'

# GPT2 small : Stanford

python translation_task.py \
    --model_name_or_path='/home/models/gpt2-translation-small-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/translation/fine_tuning_13/small_test_stanford/expansions.json' \
    --dict_file='output/translation/fine_tuning_13/small_test_stanford/dict.json' \
    --report_file='output/translation/fine_tuning_13/small_test_stanford/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='translation_small_stanford.log'
