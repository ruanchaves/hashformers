python beamsearch.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --gpu_batch_size=200 \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Dev-BOUN' \
    --eval_dataset_format='BOUN' \
    --topk=20 \
    --steps=5