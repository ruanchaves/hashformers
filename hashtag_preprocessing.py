# Script for ENACOMP 2020
import os
import pandas as pd
import re
import pathlib
hashtags_folder = '/run/media/user/DADOS/NLP/datasets/hashtags'
chunk_size = 1000

def main():
    for item in os.listdir(hashtags_folder):
        path = os.path.join(hashtags_folder,item)
        timestamp_pattern = 'hashtags_(.*?).csv'
        
        try:
            timestamp = re.search(timestamp_pattern, path).group(1)
        except:
            continue

        try:
            df = pd.read_csv(path)
            df = df.drop_duplicates('hashtag')
            df = df[['tweet_lang', 'hashtag']]
        except:
            continue

        for lang in set(df['tweet_lang']):
            lang_df = df[df['tweet_lang']==lang]
            lang_df_hashtags = df['hashtag'].values.tolist()
            lang_df_hashtags_sublist = [lang_df_hashtags[i:i+chunk_size] \
                for i in range(0, len(lang_df_hashtags), chunk_size)]
            for idx,chunk in enumerate(lang_df_hashtags_sublist):
                target_filename = 'hashtags_{0}_{1}_part_{2}.csv'.format(lang, timestamp, idx)
                target_filename = pathlib.Path(target_filename).stem + '.txt'
                target_filename = os.path.join(hashtags_folder, target_filename)
                open(target_filename,'w+').close()
                with open(target_filename,'a+') as f:
                    for individual_hashtag in chunk:
                        print(individual_hashtag, file=f)

if __name__ == '__main__':
    main()