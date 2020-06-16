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
    --dict_file="output/gpt2_small/$DATASET/steps_$STEPS/topk_$TOPK/dict.json" \
    --report_file="output/gpt2_small/$DATASET/steps_$STEPS/topk_$TOPK/report.json" \
    --topk=$TOPK \
    --steps=$STEPS \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=3000000000 \
    --logfile="gpt2_small&topk_$TOPK&steps_$STEPS&$LOG.log"

python beamsearch_manager.py \
    --model_name_or_path="gpt2-medium" \
    --model_type="gpt2" \
    --gpu_batch_size=1 \
    --eval_data_file="/home/datasets/word_segmentation/doval/$DATASET.tsv" \
    --eval_dataset_format="doval" \
    --dict_file="output/gpt2_medium/$DATASET/steps_$STEPS/topk_$TOPK/dict.json" \
    --report_file="output/gpt2_medium/$DATASET/steps_$STEPS/topk_$TOPK/report.json" \
    --topk=$TOPK \
    --steps=$STEPS \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=6000000000 \
    --logfile="gpt2_medium&topk_$TOPK&steps_$STEPS&$LOG.log"

python beamsearch_manager.py \
    --model_name_or_path="gpt2-large" \
    --model_type="gpt2" \
    --gpu_batch_size=1 \
    --eval_data_file="/home/datasets/word_segmentation/doval/$DATASET.tsv" \
    --eval_dataset_format="doval" \
    --dict_file="output/gpt2_large/$DATASET/steps_$STEPS/topk_$TOPK/dict.json" \
    --report_file="output/gpt2_large/$DATASET/steps_$STEPS/topk_$TOPK/report.json" \
    --topk=$TOPK \
    --steps=$STEPS \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=9000000000 \
    --logfile="gpt2_large&topk_$TOPK&steps_$STEPS&$LOG.log"

python beamsearch_manager.py \
    --model_name_or_path="gpt2-xl" \
    --model_type="gpt2" \
    --gpu_batch_size=1 \
    --eval_data_file="/home/datasets/word_segmentation/doval/$DATASET.tsv" \
    --eval_dataset_format="doval" \
    --dict_file="output/gpt2_xl/$DATASET/steps_$STEPS/topk_$TOPK/dict.json" \
    --report_file="output/gpt2_xl/$DATASET/steps_$STEPS/topk_$TOPK/report.json" \
    --topk=$TOPK \
    --steps=$STEPS \
    --topn=4 \
    --gpu_expansion_batch_size=50 \
    --expected_worker_load=15000000000 \
    --logfile="gpt2_xl&topk_$TOPK&steps_$STEPS&$LOG.log"

