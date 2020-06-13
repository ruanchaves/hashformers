#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..
cd ..


## Stanford

# GPT2 small : Stanford

python ranking.py \
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
    --logfile='small_stanford.log' &

## Test-BOUN

# GPT2 small : Test-BOUN

python ranking.py \
    --model_name_or_path='/home/models/gpt2-small-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/fine_tuning_13/small_test_boun/expansions.json' \
    --dict_file='output/fine_tuning_13/small_test_boun/dict.json' \
    --report_file='output/fine_tuning_13/small_test_boun/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_test_boun.log' &

## Stanford

# GPT2 small : Stanford

python ranking.py \
    --model_name_or_path='/home/models/gpt2-small-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/fine_tuning_13/small_test_stanford/expansions.json' \
    --dict_file='output/fine_tuning_13/small_test_stanford/dict.json' \
    --report_file='output/fine_tuning_13/small_test_stanford/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_stanford.log' &

## Test-BOUN

# GPT2 small : Test-BOUN

python ranking.py \
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
    --logfile='small_test_boun.log' &

## Test-BOUN

# GPT2 small : Test-BOUN

python ranking.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/small_test_boun/expansions.json' \
    --dict_file='output/small_test_boun/dict.json' \
    --report_file='output/small_test_boun/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_test_boun.log' &





## glushkova (rus)

# GPT2 small : glushkova (rus)

python ranking.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_rus.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/small_glushkova_rus/expansions.json' \
    --dict_file='output/small_glushkova_rus/dict.json' \
    --report_file='output/small_glushkova_rus/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_glushkova_rus.log' &


## glushkova (eng)

# GPT2 small : glushkova (eng)

python ranking.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_eng.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/small_glushkova_eng/expansions.json' \
    --dict_file='output/small_glushkova_eng/dict.json' \
    --report_file='output/small_glushkova_eng/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_glushkova_eng.log' &


## Stanford

# GPT2 small : Stanford

python ranking.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/small_test_stanford/expansions.json' \
    --dict_file='output/small_test_stanford/dict.json' \
    --report_file='output/small_test_stanford/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_stanford.log'

## glushkova (rus)

# GPT2 small : glushkova (rus)

python ranking.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_rus.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/small_glushkova_rus/expansions.json' \
    --dict_file='output/small_glushkova_rus/dict.json' \
    --report_file='output/small_glushkova_rus/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_glushkova_rus.log'

## glushkova (eng)

# GPT2 small : glushkova (eng)

python ranking.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_eng.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/small_glushkova_eng/expansions.json' \
    --dict_file='output/small_glushkova_eng/dict.json' \
    --report_file='output/small_glushkova_eng/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_glushkova_eng.log'

## Test-BOUN

# GPT2 small : Test-BOUN

python ranking.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/small_test_boun/expansions.json' \
    --dict_file='output/small_test_boun/dict.json' \
    --report_file='output/small_test_boun/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_test_boun.log'

## Stanford

# GPT2 small : Stanford

python ranking.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/small_test_stanford/expansions.json' \
    --dict_file='output/small_test_stanford/dict.json' \
    --report_file='output/small_test_stanford/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_stanford.log'

## glushkova (eng)

# GPT2 small : glushkova (eng)

python ranking.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_eng.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/small_glushkova_eng/expansions.json' \
    --dict_file='output/small_glushkova_eng/dict.json' \
    --report_file='output/small_glushkova_eng/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_glushkova_eng.log'

## glushkova (rus)

# GPT2 small : glushkova (rus)

python ranking.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_rus.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/small_glushkova_rus/expansions.json' \
    --dict_file='output/small_glushkova_rus/dict.json' \
    --report_file='output/small_glushkova_rus/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_glushkova_rus.log'

## Stanford

# GPT2 small : Stanford

python ranking.py \
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

## Test-BOUN

# GPT2 small : Test-BOUN

python ranking.py \
    --model_name_or_path='/home/models/gpt2-small-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/fine_tuning_13/small_test_boun/expansions.json' \
    --dict_file='output/fine_tuning_13/small_test_boun/dict.json' \
    --report_file='output/fine_tuning_13/small_test_boun/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_test_boun.log'

## Stanford

# GPT2 small : Stanford

python ranking.py \
    --model_name_or_path='/home/models/gpt2-small-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/fine_tuning_13/small_test_stanford/expansions.json' \
    --dict_file='output/fine_tuning_13/small_test_stanford/dict.json' \
    --report_file='output/fine_tuning_13/small_test_stanford/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_stanford.log'

## Test-BOUN

# GPT2 small : Test-BOUN

python ranking.py \
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

## Test-BOUN

# GPT2 small : Test-BOUN

python ranking.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/small_test_boun/expansions.json' \
    --dict_file='output/small_test_boun/dict.json' \
    --report_file='output/small_test_boun/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile='small_test_boun.log'