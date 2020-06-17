import os
from reader import DatasetReader
import json 
import pandas as pd
import numpy as np
from metrics import calculate_f1, calculate_precision, calculate_recall
def read_json(filename):
    with open(filename,'r') as f:
        return json.load(f)

def read_dataset_dir(allowed_types, dataset_dict, type_dict, OUTPUT_DIR, DATASET_DIR, dict_identifier='dict.json'):
    output = []
    for path, subpaths, files in os.walk(OUTPUT_DIR):
        for filename in files:
            if filename.endswith(dict_identifier):
                row = {}
                full_path = os.path.join(path, filename)
                test_dataset_type = [ x for x in allowed_types if x.lower() in full_path][0]
                test_dataset_name = [ x for x in dataset_dict.keys() if x.lower() in full_path][0]
                test_dataset_path = os.path.join(DATASET_DIR, dataset_dict[test_dataset_name])
                test_dataset = DatasetReader(test_dataset_path, type_dict[test_dataset_type])
                test_dataset.read()
                row['gold'] = test_dataset.test
                row['eval'] = read_json(full_path)
                row['full_path'] = full_path
                row['test_dataset_type'] = test_dataset_type
                row['test_dataset_name'] = test_dataset_name
                row['test_dataset_path'] = test_dataset_path
                output.append(row)
    return output

def build_evaluation_pairs(dict_file, gold_list):
    gold_list = [ x for x in gold_list if x.strip() ]
    gold_list = [ x.strip() for x in gold_list ]
    gold_df = pd.DataFrame(gold_list)
    gold_df = gold_df.rename(columns={
        0: 'gold'
    })
    print(max([ len(x.split()) for x in gold_df['gold'].values.tolist()]))
    gold_df['characters'] = gold_df['gold'].str.replace(" ", "")
    assert(gold_df.shape == gold_df.dropna().shape)


    test_df = [ {'hypothesis': k, 'value': v} for k,v in dict_file.items() ]
    test_df = pd.DataFrame(test_df)
    test_df['characters'] = test_df['hypothesis'].str.replace(" ", "")
    assert(test_df.shape == test_df.dropna().shape)

    eval_df = pd.merge(gold_df, test_df, on='characters', how='inner')
    assert( eval_df.shape == eval_df.dropna().shape )

    eval_df = eval_df.sort_values('value',ascending=True).groupby('gold', sort=False).head(1)

    return eval_df

def generate_metrics(df, target=None):
    df['f1'] = df['gold'].combine(df['hypothesis'], calculate_f1)
    df['precision'] = df['gold'].combine(df['hypothesis'], calculate_precision)
    df['recall'] = df['gold'].combine(df['hypothesis'], calculate_recall)
    if target:
        df.to_csv(target)
    df = df[['f1', 'precision', 'recall']]
    df = df.agg([np.mean, np.std])
    return df

def main():
    allowed_types = [
        "glushkova",
        "BOUN",
        "stanford",
        "de_news"
    ]
    dataset_dict = {
        "glushkova_eng": "hashtag_segmentation/glushkova/test_eng.csv",
        "glushkova_rus": "hashtag_segmentation/glushkova/test_rus.csv",
        "BOUN": "hashtag_segmentation/BOUN/Test-BOUN",
        "stanford": "hashtag_segmentation/Stanford/stanford_dataset.txt",
        "de_news": "word_segmentation/doval/news/de_news.tsv"
    }
    type_dict = {
        "glushkova": "glushkova",
        "BOUN": "BOUN",
        "stanford": "stanford",
        "de_news": "doval"
    }

    OUTPUT_DIR = './output'
    DATASET_DIR = "../../datasets"
    output = read_dataset_dir(allowed_types, dataset_dict, type_dict, OUTPUT_DIR, DATASET_DIR)
    result = []
    for idx, item in enumerate(output):
        try:
            print(item['full_path'])
            evaluation_df = build_evaluation_pairs(item['eval'], item['gold'])

            path_as_fname = item['full_path'].replace("/", "_")
            evaluation_df.to_csv('./logs/tmp_{0}.csv'.format(path_as_fname))
            if 'glushkova_eng' in item['full_path']:
                metrics = generate_metrics(evaluation_df, target='tmp.csv')
            else:
                metrics = generate_metrics(evaluation_df)
            print(metrics)
        
        metrics = metrics.reset_index()
        metrics = metrics.to_dict('records')
        print(metrics)

        except Exception as e:
            print(e, item['full_path'])

if __name__ == '__main__':
    main()