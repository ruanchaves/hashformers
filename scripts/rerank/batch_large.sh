#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..
cd ..

# GPT2 large : Stanford

python ranking.py \
    --model_name_or_path='gpt2-large' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/large_test_stanford/expansions.json' \
    --dict_file='output/large_test_stanford/dict.json' \
    --report_file='output/large_test_stanford/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_stanford.log'

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
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_glushkova_rus.log'

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
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_glushkova_eng.log'

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
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_test_boun.log'

# GPT2 large : Stanford

python ranking.py \
    --model_name_or_path='gpt2-large' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/large_test_stanford/expansions.json' \
    --dict_file='output/large_test_stanford/dict.json' \
    --report_file='output/large_test_stanford/report.json' \
    --topk=20 \
    --steps=13 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=12000000000 \
    --logfile='large_stanford.log'

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