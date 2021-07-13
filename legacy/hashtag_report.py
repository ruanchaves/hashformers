import os
import pandas as pd  
import pathlib 
import json

path = 'output/enacomp2020'
target_file = 'enacomp2020_report.txt'

def os_walk(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith('.json'):
                yield root, name

def main():
    dct = {}
    for root, name in os_walk(path):
        source_path = os.path.join(root, name)
        with open(source_path,'r') as f:
            source_dct = json.load(f)
        dct.update(source_dct)

    df = [ {'key': k, 'value': v} for k,v in dct.items() ]
    df = pd.DataFrame(df)
    df['raw'] = df['key'].apply( lambda x: x.replace(" ", "") )
    df = df.sort_values('value').groupby('raw').head(1)
    hashtags = df['key'].values.tolist()
    open(target_file,'w+').close()
    with open(target_file,'a+') as f:
        for item in hashtags:
            print(item, file=f)

if __name__ == '__main__':
    main()