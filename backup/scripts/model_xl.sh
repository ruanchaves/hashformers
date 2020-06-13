#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..

## Test-BOUN

# GPT2 extra-large : Test-BOUN

python beamsearch.py \
    --model_name_or_path='gpt2-xl' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/xl_test_boun/expansions.json' \
    --dict_file='output/xl_test_boun/dict.json' \
    --report_file='output/xl_test_boun/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --logfile='xl_test_boun.log'

## glushkova (eng)

# GPT2 extra-large : glushkova (eng)

python beamsearch.py \
    --model_name_or_path='gpt2-xl' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_eng.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/xl_glushkova_eng/expansions.json' \
    --dict_file='output/xl_glushkova_eng/dict.json' \
    --report_file='output/xl_glushkova_eng/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --logfile='xl_glushkova_eng.log'

## glushkova (rus)

# GPT2 extra-large : glushkova (rus)

python beamsearch.py \
    --model_name_or_path='gpt2-xl' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/glushkova/test_rus.csv' \
    --eval_dataset_format='glushkova' \
    --expansions_file='output/xl_glushkova_rus/expansions.json' \
    --dict_file='output/xl_glushkova_rus/dict.json' \
    --report_file='output/xl_glushkova_rus/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --logfile='xl_glushkova_rus.log'


## Stanford

# GPT2 extra-large : Stanford

python beamsearch.py \
    --model_name_or_path='gpt2-xl' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file='/home/datasets/hashtag_segmentation/Stanford/stanford_dataset.txt' \
    --eval_dataset_format='stanford' \
    --expansions_file='output/xl_test_stanford/expansions.json' \
    --dict_file='output/xl_test_stanford/dict.json' \
    --report_file='output/xl_test_stanford/report.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --logfile='xl_stanford.log'