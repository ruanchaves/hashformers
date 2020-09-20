# Script for ENACOMP 2020
import os
import pandas as pd
import re
import pathlib
hashtags_folder = '/run/media/user/DADOS/NLP/datasets/hashtags'
chunk_size = 1000

def main():
    df_list = []
    for item in os.listdir(hashtags_folder):
        path = os.path.join(hashtags_folder,item)
        timestamp_pattern = 'hashtags_(.*?).csv'
        
        try:
            timestamp = re.search(timestamp_pattern, path).group(1)
        except:
            continue

        try:
            df = pd.read_csv(path)
            df['timestamp'] = timestamp
            df = df[['timestamp', 'tweet_lang', 'hashtag']]
        except:
            continue

        df_list.append(df)

    df = pd.concat(df_list)

    df = df.drop_duplicates(subset='hashtag')

    visited_hashtags = {}

    for lang in set(df['tweet_lang']):
        lang_df = df[df['tweet_lang']==lang]
        for timestamp in set(lang_df['timestamp']):
            timestamp_df = lang_df[lang_df['timestamp']==timestamp]

            df_hashtags = timestamp_df['hashtag'].values.tolist()
            df_hashtags_sublist = [df_hashtags[i:i+chunk_size] \
                for i in range(0, len(df_hashtags), chunk_size)]
            for idx,chunk in enumerate(df_hashtags_sublist):
                target_filename = 'hashtags_{0}_{1}_part_{2}.csv'.format(lang, timestamp, idx)
                target_filename = pathlib.Path(target_filename).stem + '.txt'
                target_filename = os.path.join(hashtags_folder, target_filename)
                open(target_filename,'w+').close()
                with open(target_filename,'a+') as f:
                    for individual_hashtag in chunk:
                        visited_status = visited_hashtags.get(individual_hashtag, False)
                        if visited_status:
                            continue
                        else:
                            print(individual_hashtag, file=f)
                            visited_hashtags[individual_hashtag] = True

if __name__ == '__main__':
    main()