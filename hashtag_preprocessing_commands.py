# Script for ENACOMP 2020
import os
import pandas as pd
import re
import pathlib
hashtags_folder = '/run/media/user/DADOS/NLP/datasets/hashtags'
script_destination = 'scripts/enacomp2020.sh'

def main():
    command = """
    python beamsearch_manager.py \\
        --model_name_or_path='{0}' \\
        --model_type='gpt2' \\
        --gpu_batch_size=1 \\
        --eval_data_file='/home/datasets/hashtags/{1}' \\
        --eval_dataset_format='BOUN' \\
        --expansions_file='output/enacomp2020/{2}/expansions.json' \\
        --dict_file='output/enacomp2020/{2}/dict.json' \\
        --report_file='output/enacomp2020/{2}/report.json' \\
        --topk=20 \\
        --steps=5 \\
        --topn=4 \\
        --gpu_expansion_batch_size=50 \\
        --expected_worker_load=3000000000 \\
        --logfile='output/enacomp2020/{2}/logfile.log'

    """

    filename_pattern = 'hashtags_(.*?).txt'
    open(script_destination,'w+').close()

    en_cmds = []
    pt_cmds = []
    for filename in os.listdir(hashtags_folder):
        try:
            _ = re.search(filename_pattern, filename).group(1)
        except:
            continue
        
        folder_name = pathlib.Path(filename).stem
        if 'en' in filename:
            row = command.format('gpt2', filename, folder_name)
            en_cmds.append(row)
        elif 'pt' in filename:
            row = command.format('pierreguillou/gpt2-small-portuguese', filename, folder_name)
            pt_cmds.append(row)

    cmds = en_cmds + pt_cmds
    with open(script_destination,'a+') as f:
        for row in cmds:
            print(row, file=f)

if __name__ == '__main__':
    main()