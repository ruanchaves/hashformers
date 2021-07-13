#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..

## Test-BOUN

# GPT2 small : Test-BOUN

python beamsearch_manager.py \
    --model_name_or_path='/home/models/gpt2-small-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/fine_tuning/small_test_boun/expansions.json' \
    --dict_file='output/fine_tuning/small_test_boun/dict.json' \
    --report_file='output/fine_tuning/small_test_boun/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_test_boun.log'

# GPT2 medium : Test-BOUN

python beamsearch_manager.py \
    --model_name_or_path='/home/models/gpt2-medium-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/fine_tuning/medium_test_boun/expansions.json' \
    --dict_file='output/fine_tuning/medium_test_boun/dict.json' \
    --report_file='output/fine_tuning/medium_test_boun/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=6000000000 \
    --logfile='medium_test_boun.log'

# GPT2 large : Test-BOUN

python beamsearch_manager.py \
    --model_name_or_path='/home/models/gpt2-large-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/fine_tuning/large_test_boun/expansions.json' \
    --dict_file='output/fine_tuning/large_test_boun/dict.json' \
    --report_file='output/fine_tuning/large_test_boun/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_test_boun.log'

## Stanford

# GPT2 small : Stanford

python beamsearch_manager.py \
    --model_name_or_path='/home/models/gpt2-small-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/fine_tuning/small_test_stanford/expansions.json' \
    --dict_file='output/fine_tuning/small_test_stanford/dict.json' \
    --report_file='output/fine_tuning/small_test_stanford/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_stanford.log'

# GPT2 medium : Stanford

python beamsearch_manager.py \
    --model_name_or_path='/home/models/gpt2-medium-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/fine_tuning/medium_test_stanford/expansions.json' \
    --dict_file='output/fine_tuning/medium_test_stanford/dict.json' \
    --report_file='output/fine_tuning/medium_test_stanford/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=6000000000 \
    --logfile='medium_stanford.log'

# GPT2 large : Stanford

python beamsearch_manager.py \
    --model_name_or_path='/home/models/gpt2-large-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/fine_tuning/large_test_stanford/expansions.json' \
    --dict_file='output/fine_tuning/large_test_stanford/dict.json' \
    --report_file='output/fine_tuning/large_test_stanford/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_stanford.log'
