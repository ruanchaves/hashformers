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

def build_evaluation_pairs(dict_file, gold_list, curve_range=100):
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

    def best_n(df, n=1):
        tmp = df.sort_values('value', ascending=True).groupby('gold', sort=False).head(n)
        df_size = df.drop_duplicates('gold')
        tmp = tmp[tmp['hypothesis']==tmp['gold']].drop_duplicates('gold')
        return tmp.shape[0] / df_size.shape[0]
    
    curve = []
    for i in range(curve_range):
        try:
            curve.append(best_n(eval_df, n=i))
        except:
            continue

    eval_df = eval_df.sort_values('value',ascending=True).groupby('gold', sort=False).head(1)
    return curve, eval_df

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
        "de_news": "word_segmentation/doval/news/de_news.tsv",
        "en_news": "word_segmentation/doval/news/en_news.tsv",
        "fi_news": "word_segmentation/doval/news/fi_news.tsv",
        "tr_news": "word_segmentation/doval/news/tr_news.tsv"
    }
    type_dict = {
        "glushkova": "glushkova",
        "BOUN": "BOUN",
        "stanford": "stanford",
        "de_news": "doval",
        "en_news": "doval",
        "fi_news": "doval",
        "tr_news": "doval"
    }

    OUTPUT_DIR = './output'
    DATASET_DIR = "../../datasets"
    output = read_dataset_dir(allowed_types, dataset_dict, type_dict, OUTPUT_DIR, DATASET_DIR)
    result = []
    for idx, item in enumerate(output):
        try:
            print(item['full_path'])
            curve, evaluation_df = build_evaluation_pairs(item['eval'], item['gold'])
            path_as_fname = item['full_path'].replace("/", "_")
            metrics = generate_metrics(evaluation_df)
            metrics = metrics.reset_index()
            metrics = metrics.to_dict('records')
            metrics = [ x for x in metrics if x['index']=='mean']
            metrics[0]['full_path'] = item['full_path']
            metrics[0]['curve'] = curve
            result.append(metrics[0])
        except Exception as e:
            print(e, item['full_path'])
    pd.DataFrame(result).to_csv('results.csv')

if __name__ == '__main__':
    main()