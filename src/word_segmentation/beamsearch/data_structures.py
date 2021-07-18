import pandas as pd
from dataclasses import dataclass
import json 
import numpy as np 

@dataclass
class Node:
    hypothesis: str
    characters: str
    score: float

@dataclass
class ProbabilityDictionary(object):
    dictionary: dict

    def get_segmentations(
        self,
        astype='dict',
        gold_array=None
    ):
        top_1 = self.get_top_k(k=1)
        if gold_array and astype == 'list':
            gold_df = pd.DataFrame([{
                "gold": x,
                "characters": x.replace(" ", "")
            } for x in gold_array])
            seg_df = pd.DataFrame([{
                "segmentation": x,
                "characters": x.replace(" ", "") 
            } for x in top_1])
            output_df = pd.merge(
                gold_df,
                seg_df,
                how='left',
                on='characters'
            )
            output_series = output_df['segmentation'].values.tolist()
            output_series = [
                str(x) for x in output_series
            ]
            return output_series
        if astype == 'dict':
            return { k.replace(" ", ""):k for k,v in top_1.items() }
        elif astype == 'list':
            return list(top_1.keys())

    def get_top_k(
        self,
        k=2,
        characters_field='characters',
        segmentation_field='segmentation',
        score_field='score',
        return_dataframe=False,
        fill=False
    ):
        df = self.to_dataframe(
            characters_field=characters_field,
            segmentation_field=segmentation_field,
            score_field=score_field
        )
        df = df\
        .sort_values(by=score_field, ascending=True)\
        .groupby(characters_field)\
        .head(k)
        if fill == False and return_dataframe == True:
            return df
        elif fill == True and return_dataframe == True:
            df['group_length'] = df.groupby(characters_field)[segmentation_field].transform(len)
            df['group_length'] = df['group_length'] * -1 + k + 1
            len_array = df['group_length'].values            
            df = df.drop(columns=['group_length'])
            records = np.array(df.to_dict('records'))
            cloned_records = list(np.repeat(records, len_array))
            df = pd.DataFrame(cloned_records)
            return df
        elif fill == False and return_dataframe == False:
            keys = df[segmentation_field].values
            values = df[score_field].values
            output = {
                k:v for k,v in list(zip(keys, values))
            }
            return output
        elif fill == True and return_dataframe == False:
            raise NotImplementedError

    def to_dataframe(
            self,
            characters_field='characters',
            segmentation_field='segmentation',
            score_field='score'):
        df = [
            {
                characters_field: key.replace(" ", ""),
                segmentation_field: key,
                score_field: value
            } for key, value \
                in self.dictionary.items()
        ]
        df = pd.DataFrame(df)
        df = df.sort_values(
            by=[
                characters_field, 
                score_field
            ]
        )
        return df
    
    def to_csv(
        self,
        filename,
        characters_field='characters',
        segmentation_field='segmentation',
        score_field='score'
    ):
        df = self.to_dataframe(
            characters_field=characters_field,
            segmentation_field=segmentation_field,
            score_field=score_field
        )
        df.to_csv(filename)


    def to_json(
        self,
        filepath
    ):
        with open(filepath, 'w') as f:
            json.dump(self.dictionary, f)

def enforce_prob_dict(
    dictionary,
    score_field="score",
    segmentation_field="segmentation"):
    if isinstance(dictionary, ProbabilityDictionary):
        return dictionary
    elif isinstance(dictionary, dict):
        return ProbabilityDictionary(dictionary)
    elif isinstance(dictionary, list) \
        and all(isinstance(x, str) for x in dictionary):
        dct = {
            k:0.0 for k in list(set(dictionary))
        }
        return ProbabilityDictionary(dct)
    elif isinstance(dictionary, pd.DataFrame):
        df = dictionary
        df_scores = df[score_field].values.tolist()
        df_segs = df[segmentation_field].values.tolist()
        dct = {
            k:v for k,v in list(zip(df_segs, df_scores))
        }
        return ProbabilityDictionary(dct)
    else:
        raise NotImplementedError