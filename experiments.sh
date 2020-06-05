#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

pip install -r requirements.txt

# Test run
VALIDATION_SET='/home/datasets/hashtag_segmentation/gpt2/validation' \
TRAINING_SET_1='/home/datasets/hashtag_segmentation/gpt2/artificial_1' \
TRAINING_SET_2='/home/datasets/hashtag_segmentation/gpt2/artificial_2' \
EVALUATION_SET='/home/datasets/hashtag_segmentation/gpt2/natural' \
CNN_SAVE_PATH='/home/models/gpt2_cnn/test_run/cnn_model.pth' \
MLP_SAVE_PATH='/home/models/gpt2_mlp/test_run/mlp_model.pth' \
bash model.sh

VALIDATION_SET='/home/datasets/hashtag_segmentation/BOUN/Dev-BOUN_and_Dev-Stanford' \
TRAINING_SET_1='/home/datasets/hashtag_segmentation/BOUN/sample_100_clean.SNAP.Hashtags.Segmented.w.Heuristics_1' \
TRAINING_SET_2='/home/datasets/hashtag_segmentation/BOUN/sample_100_clean.SNAP.Hashtags.Segmented.w.Heuristics_2' \
EVALUATION_SET='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
CNN_SAVE_PATH='/home/models/gpt2_cnn/sample_100_snap/cnn_model.pth' \
MLP_SAVE_PATH='/home/models/gpt2_mlp/sample_100_snap/mlp_model.pth' \
bash model.sh

VALIDATION_SET='/home/datasets/hashtag_segmentation/BOUN/Dev-BOUN_and_Dev-Stanford' \
TRAINING_SET_1='/home/datasets/hashtag_segmentation/BOUN/sample_500_clean.SNAP.Hashtags.Segmented.w.Heuristics_1' \
TRAINING_SET_2='/home/datasets/hashtag_segmentation/BOUN/sample_500_clean.SNAP.Hashtags.Segmented.w.Heuristics_2' \
EVALUATION_SET='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
CNN_SAVE_PATH='/home/models/gpt2_cnn/sample_500_snap/cnn_model.pth' \
MLP_SAVE_PATH='/home/models/gpt2_mlp/sample_500_snap/mlp_model.pth' \
bash model.sh

VALIDATION_SET='/home/datasets/hashtag_segmentation/BOUN/Dev-BOUN_and_Dev-Stanford' \
TRAINING_SET_1='/home/datasets/hashtag_segmentation/BOUN/sample_1000_clean.SNAP.Hashtags.Segmented.w.Heuristics_1' \
TRAINING_SET_2='/home/datasets/hashtag_segmentation/BOUN/sample_1000_clean.SNAP.Hashtags.Segmented.w.Heuristics_2' \
EVALUATION_SET='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
CNN_SAVE_PATH='/home/models/gpt2_cnn/sample_1000_snap/cnn_model.pth' \
MLP_SAVE_PATH='/home/models/gpt2_mlp/sample_1000_snap/mlp_model.pth' \
bash model.sh

VALIDATION_SET='/home/datasets/hashtag_segmentation/BOUN/Dev-BOUN_and_Dev-Stanford' \
TRAINING_SET_1='/home/datasets/hashtag_segmentation/BOUN/sample_5000_clean.SNAP.Hashtags.Segmented.w.Heuristics_1' \
TRAINING_SET_2='/home/datasets/hashtag_segmentation/BOUN/sample_5000_clean.SNAP.Hashtags.Segmented.w.Heuristics_2' \
EVALUATION_SET='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
CNN_SAVE_PATH='/home/models/gpt2_cnn/sample_5000_snap/cnn_model.pth' \
MLP_SAVE_PATH='/home/models/gpt2_mlp/sample_5000_snap/mlp_model.pth' \
bash model.sh

VALIDATION_SET='/home/datasets/hashtag_segmentation/BOUN/Dev-BOUN_and_Dev-Stanford' \
TRAINING_SET_1='/home/datasets/hashtag_segmentation/BOUN/sample_10000_clean.SNAP.Hashtags.Segmented.w.Heuristics_1' \
TRAINING_SET_2='/home/datasets/hashtag_segmentation/BOUN/sample_10000_clean.SNAP.Hashtags.Segmented.w.Heuristics_2' \
EVALUATION_SET='/home/datasets/hashtag_segmentation/BOUN/Test-BOUN' \
CNN_SAVE_PATH='/home/models/gpt2_cnn/sample_10000_snap/cnn_model.pth' \
MLP_SAVE_PATH='/home/models/gpt2_mlp/sample_10000_snap/mlp_model.pth' \
bash model.sh