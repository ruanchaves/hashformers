#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

# Validation set

python beamsearch_manager.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file=${VALIDATION_SET} \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/expansions_validation.json' \
    --dict_file='output/dict_validation.json' \
    --report_file='output/report_validation.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50

# CNN model training

python beamsearch_manager.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file=${TRAINING_SET_1} \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/expansions_artificial_cnn.json' \
    --dict_file='output/dict_artificial_cnn.json' \
    --report_file='output/report_artificial_cnn.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50

python cnn_model.py \
    --model_name_or_path='gpt2' \
    --expansions_file='output/expansions_artificial_cnn.json' \
    --dict_file='output/dict_artificial_cnn.json' \
    --report_file='output/report_artificial_cnn.json' \
    --validation_expansions_file='output/expansions_validation.json' \
    --validation_dict_file='output/dict_validation.json' \
    --validation_report_file='output/report_validation.json' \
    --filter_sizes='3,4,5' \
    --output-dim=1 \
    --dropout=0.5 \
    --cnn_device='cuda' \
    --token_embedding_size=768 \
    --n_filters=768 \
    --cnn_learning_rate=0.0001 \
    --cnn_training_epochs=100 \
    --cnn_missed_epoch_limit=10 \
    --cnn_save_path=${CNN_SAVE_PATH} \
    --cnn_batch_size=${CNN_BATCH_SIZE}

# MLP model training

python beamsearch_manager.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file=${TRAINING_SET_2} \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/expansions_artificial_mlp.json' \
    --dict_file='output/dict_artificial_mlp.json' \
    --report_file='output/report_artificial_mlp.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50

python mlp_model.py \
    --model_name_or_path='gpt2' \
    --expansions_file='output/expansions_artificial_mlp.json' \
    --dict_file='output/dict_artificial_mlp.json' \
    --validation_expansions_file='output/expansions_validation.json' \
    --validation_dict_file='output/dict_validation.json' \
    --validation_report_file='output/report_validation.json' \
    --report_file='output/report_artificial_mlp.json' \
    --mlp_device='cuda' \
    --mlp_training_epochs=100 \
    --mlp_missed_epoch_limit=10 \
    --mlp_learning_rate=0.001 \
    --cnn_save_path=${CNN_SAVE_PATH} \
    --mlp_save_path=${MLP_SAVE_PATH} \
    --mlp_batch_size=${MLP_BATCH_SIZE}

# GPT2 -> CNN -> MLP evaluation

python beamsearch_manager.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=1 \
    --eval_data_file=${EVALUATION_SET} \
    --eval_dataset_format='BOUN' \
    --expansions_file='output/expansions_natural.json' \
    --dict_file='output/dict_natural.json' \
    --report_file='output/report_natural.json' \
    --topk=20 \
    --steps=5 \
    --topn=4 \
    --gpu_expansion_batch_size=50

python model_evaluation.py \
    --model_name_or_path='gpt2' \
    --expansions_file='output/expansions_natural.json' \
    --dict_file='output/dict_natural.json' \
    --report_file='output/report_natural.json' \
    --cnn_save_path=${CNN_SAVE_PATH} \
    --mlp_save_path=${MLP_SAVE_PATH}