#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
cd ..

python dataset_translation_format.py /home/datasets/hashtag_segmentation/BOUN/clean.SNAP.Hashtags.Segmented.w.Heuristics /home/datasets/hashtag_segmentation/BOUN/translate_task_snap
python dataset_translation_format /home/datasets/hashtag_segmentation/BOUN/Dev-BOUN_and_Dev-Stanford /home/datasets/hashtag_segmentation/BOUN/translate_task_dev

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python run_language_modeling.py \
    --output_dir='/home/models/gpt2-translation-small-snap' \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --save_total_limit=5 \
    --num_train_epochs=3.0 \
    --do_train \
    --evaluate_during_training \
    --logging_steps=500 \
    --save_steps=500 \
    --train_data_file='/home/datasets/hashtag_segmentation/BOUN/clean.SNAP.Hashtags.Segmented.w.Heuristics /home/datasets/hashtag_segmentation/BOUN/translate_task_snap' \
    --do_eval \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/translate_task_dev' \
    --per_gpu_train_batch_size=70 \
    --per_gpu_eval_batch_size=70 \
    --block_size=128 \
    --gradient_accumulation_steps=1

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"
