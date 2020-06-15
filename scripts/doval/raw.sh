#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..
cd ..

python beamsearch_manager.py \
    --model_name_or_path="gpt2" \
    --model_type="gpt2" \
    --gpu_batch_size=1 \
    --eval_data_file="/home/datasets/word_segmentation/doval/$DATASET.tsv" \
    --eval_dataset_format="doval" \
    --dict_file="output/gpt2_small/$DATASET/dict.json" \
    --report_file="output/gpt2_small/$DATASET/report.json" \
    --topk=20 \
    --steps=$STEPS \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile="gpt2_small_$LOG.log"

python beamsearch_manager.py \
    --model_name_or_path="gpt2-medium" \
    --model_type="gpt2" \
    --gpu_batch_size=1 \
    --eval_data_file="/home/datasets/word_segmentation/doval/$DATASET.tsv" \
    --eval_dataset_format="doval" \
    --dict_file="output/gpt2_medium/$DATASET/dict.json" \
    --report_file="output/gpt2_medium/$DATASET/report.json" \
    --topk=20 \
    --steps=$STEPS \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=6000000000 \
    --logfile="gpt2_medium_$LOG.log"

python beamsearch_manager.py \
    --model_name_or_path="gpt2-large" \
    --model_type="gpt2" \
    --gpu_batch_size=1 \
    --eval_data_file="/home/datasets/word_segmentation/doval/$DATASET.tsv" \
    --eval_dataset_format="doval" \
    --dict_file="output/gpt2_large/$DATASET/dict.json" \
    --report_file="output/gpt2_large/$DATASET/report.json" \
    --topk=20 \
    --steps=$STEPS \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=9000000000 \
    --logfile="gpt2_large_$LOG.log"

python beamsearch_manager.py \
    --model_name_or_path="gpt2-xl" \
    --model_type="gpt2" \
    --gpu_batch_size=1 \
    --eval_data_file="/home/datasets/word_segmentation/doval/$DATASET.tsv" \
    --eval_dataset_format="doval" \
    --dict_file="output/gpt2_xl/$DATASET/dict.json" \
    --report_file="output/gpt2_xl/$DATASET/report.json" \
    --topk=20 \
    --steps=$STEPS \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=15000000000 \
    --logfile="gpt2_xl_$LOG.log"

