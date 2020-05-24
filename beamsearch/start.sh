python beamsearch.py \
    --model_name_or_path='gpt2' \
    --model_type='gpt2' \
    --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/Dev-BOUN' \
    --eval_dataset_format='BOUN' \
    --gpu_batch_size=200 \
    --topk=10 \
    --steps=5