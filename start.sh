# python beamsearch.py \
#     --model_name_or_path='gpt2' \
#     --model_type='gpt2' \
#     --gpu_batch_size=1000 \
#     --eval_data_file='/home/datasets/hashtag_segmentation/BOUN/toy' \
#     --eval_dataset_format='BOUN' \
#     --expansions_file='expansions.json' \
#     --dict_file='dict.json' \
#     --report_file='report.json' \
#     --topk=20 \
#     --steps=5 \
#     --topn=4 \
#     --gpu_expansion_batch_size=50

python gpt2_encoder.py \
    --model_name_or_path='gpt2' \
    --expansions_file='expansions.json' \
    --dict_file='dict.json' \
    --report_file='report.json'

# python cnnlstm_train.py \
#     --model_name_or_path='gpt2' \
#     --expansions_file='expansions.json' \
#     --dict_file='dict.json' \
#     --report_file='report.json' \

# python cnnlstm_eval.py \
#     --cnn_save_path='' \
#     --lstm_save_path='' \
#     --expansions_file='expansions.json' \
#     --dict_file='dict.json' \
#     --report_file='report.json' \