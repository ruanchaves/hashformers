#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python run_language_modeling.py \
    --output_dir='/home/models/gpt2-large-snap' \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-large \
    --save_total_limit=5 \
    --num_train_epochs=3.0 \
    --do_train \
    --evaluate_during_training \
    --logging_steps=500 \
    --save_steps=500 \
    --train_data_file='/home/datasets/hashtag_segmentation/BOUN/clean.SNAP.Hashtags.Segmented.w.Heuristics' \
    --do_eval \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Dev-BOUN_and_Dev-Stanford' \
    --per_gpu_train_batch_size=15 \
    --per_gpu_eval_batch_size=15 \
    --block_size=128 \
    --gradient_accumulation_steps=1

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
