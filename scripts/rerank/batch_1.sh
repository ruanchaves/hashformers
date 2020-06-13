#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..
cd ..

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

# GPT2 medium : Test-BOUN

python ranking.py \
    --model_name_or_path='/home/models/gpt2-medium-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/fine_tuning_13/medium_test_boun/expansions.json' \
    --dict_file='output/fine_tuning_13/medium_test_boun/dict.json' \
    --report_file='output/fine_tuning_13/medium_test_boun/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=6000000000 \
    --logfile='medium_test_boun.log'

# GPT2 large : Test-BOUN

python ranking.py \
    --model_name_or_path='/home/models/gpt2-large-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/fine_tuning_13/large_test_boun/expansions.json' \
    --dict_file='output/fine_tuning_13/large_test_boun/dict.json' \
    --report_file='output/fine_tuning_13/large_test_boun/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_test_boun.log'

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

# GPT2 medium : Stanford

python ranking.py \
    --model_name_or_path='/home/models/gpt2-medium-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/fine_tuning_13/medium_test_stanford/expansions.json' \
    --dict_file='output/fine_tuning_13/medium_test_stanford/dict.json' \
    --report_file='output/fine_tuning_13/medium_test_stanford/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=6000000000 \
    --logfile='medium_stanford.log'

# GPT2 large : Stanford

python ranking.py \
    --model_name_or_path='/home/models/gpt2-large-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/fine_tuning_13/large_test_stanford/expansions.json' \
    --dict_file='output/fine_tuning_13/large_test_stanford/dict.json' \
    --report_file='output/fine_tuning_13/large_test_stanford/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_stanford.log'

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

# GPT2 medium : Test-BOUN

python ranking.py \
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

python ranking.py \
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

# GPT2 medium : Stanford

python ranking.py \
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

python ranking.py \
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

## Test-BOUN

# GPT2 extra-large : Test-BOUN

python ranking.py \
    --model_name_or_path='/home/models/gpt2-xl-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/fine_tuning_13/xl_test_boun/expansions.json' \
    --dict_file='output/fine_tuning_13/xl_test_boun/dict.json' \
    --report_file='output/fine_tuning_13/xl_test_boun/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --logfile='xl_test_boun.log'

## Stanford

# GPT2 extra-large : Stanford

python ranking.py \
    --model_name_or_path='/home/models/gpt2-xl-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/fine_tuning_13/xl_test_stanford/expansions.json' \
    --dict_file='output/fine_tuning_13/xl_test_stanford/dict.json' \
    --report_file='output/fine_tuning_13/xl_test_stanford/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --logfile='xl_stanford.log'

## Test-BOUN

# GPT2 extra-large : Test-BOUN

python ranking.py \
    --model_name_or_path='/home/models/gpt2-xl-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/fine_tuning/xl_test_boun/expansions.json' \
    --dict_file='output/fine_tuning/xl_test_boun/dict.json' \
    --report_file='output/fine_tuning/xl_test_boun/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --logfile='xl_test_boun.log'

## Stanford

# GPT2 extra-large : Stanford

python ranking.py \
    --model_name_or_path='/home/models/gpt2-xl-snap' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/fine_tuning/xl_test_stanford/expansions.json' \
    --dict_file='output/fine_tuning/xl_test_stanford/dict.json' \
    --report_file='output/fine_tuning/xl_test_stanford/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --logfile='xl_stanford.log'

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

# GPT2 medium : Test-BOUN

python ranking.py \
    --model_name_or_path='gpt2-medium' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/medium_test_boun/expansions.json' \
    --dict_file='output/medium_test_boun/dict.json' \
    --report_file='output/medium_test_boun/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=6000000000 \
    --logfile='medium_test_boun.log'

# GPT2 large : Test-BOUN

python ranking.py \
    --model_name_or_path='gpt2-large' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/large_test_boun/expansions.json' \
    --dict_file='output/large_test_boun/dict.json' \
    --report_file='output/large_test_boun/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_test_boun.log'

# # GPT2 extra-large : Test-BOUN

# python ranking.py \
#     --model_name_or_path='gpt2-xl' \
#     --model_type='gpt2' \
#     --gpu_batch_size=1 \
#     --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
#     --eval_dataset_format='BOUN' \
#     --expansions_file='output/xl_test_boun/expansions.json' \
#     --dict_file='output/xl_test_boun/dict.json' \
#     --report_file='output/xl_test_boun/report.json' \
#     --topk=20 \
#     --steps=13 \
#     --topn=4 \
#     --gpu_expansion_batch_size=50 \
#     --expected_worker_load=24000000000 \
#     --logfile='xl_test_boun.log'

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

# GPT2 medium : glushkova (eng)

python ranking.py \
    --model_name_or_path='gpt2-medium' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_eng.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/medium_glushkova_eng/expansions.json' \
    --dict_file='output/medium_glushkova_eng/dict.json' \
    --report_file='output/medium_glushkova_eng/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=6000000000 \
    --logfile='medium_glushkova_eng.log'

# GPT2 large : glushkova (eng)

python ranking.py \
    --model_name_or_path='gpt2-large' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_eng.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/large_glushkova_eng/expansions.json' \
    --dict_file='output/large_glushkova_eng/dict.json' \
    --report_file='output/large_glushkova_eng/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_glushkova_eng.log'

# # GPT2 extra-large : glushkova (eng)

# python ranking.py \
#     --model_name_or_path='gpt2-xl' \
#     --model_type='gpt2' \
#     --gpu_batch_size=1 \
#     --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_eng.csv' \
#     --eval_dataset_format='glushkova' \
#     --expansions_file='output/xl_glushkova_eng/expansions.json' \
#     --dict_file='output/xl_glushkova_eng/dict.json' \
#     --report_file='output/xl_glushkova_eng/report.json' \
#     --topk=20 \
#     --steps=13 \
#     --topn=4 \
#     --gpu_expansion_batch_size=50 \
#     --expected_worker_load=24000000000 \
#     --logfile='xl_glushkova_eng.log'

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

# GPT2 medium : glushkova (rus)

python ranking.py \
    --model_name_or_path='gpt2-medium' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_rus.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/medium_glushkova_rus/expansions.json' \
    --dict_file='output/medium_glushkova_rus/dict.json' \
    --report_file='output/medium_glushkova_rus/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=6000000000 \
    --logfile='medium_glushkova_rus.log'

# GPT2 large : glushkova (rus)

python ranking.py \
    --model_name_or_path='gpt2-large' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_rus.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/large_glushkova_rus/expansions.json' \
    --dict_file='output/large_glushkova_rus/dict.json' \
    --report_file='output/large_glushkova_rus/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_glushkova_rus.log'
