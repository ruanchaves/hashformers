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
        self
    ):
        top_1 = self.get_top_k(k=1)
        return { k.replace(" ", ""):k for k,v in top_1.items() }

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
        if fill == False and return_dataframe == True:
            df = df\
            .sort_values(by=score_field, ascending=True)\
            .groupby(characters_field)\
            .head(k)
            return df
        elif fill == True and return_dataframe == True:
            df['group_length'] = df.groupby(characters_field)[segmentation_field].transform(len)
            df['group_length'] = df['group_length'] * -1 + k + 1
            len_array = df['group_length'].values            
            df = df.drop(columns=['group_length'])
            records = np.array(df.to_dict('records'))
            cloned_records = list(np.repeat(records, len_array))
            df = pd.DataFrame(cloned_records)
            df = df\
            .sort_values(by=score_field, ascending=True)\
            .groupby(characters_field)\
            .head(k)
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

    def to_json(
        self,
        filepath
    ):
        with open(filepath, 'w') as f:
            json.dump(self.dictionary, f)

def enforce_prob_dict(dictionary, coerce_lists=True):
    if isinstance(dictionary, ProbabilityDictionary):
        return dictionary
    elif isinstance(dictionary, dict):
        return ProbabilityDictionary(dictionary)
    elif isinstance(dictionary, list) \
        and all(isinstance(x, str) for x in dictionary) \
        and coerce_lists == True:
        dct = {
            k:0.0 for k in list(set(dictionary))
        }
        return ProbabilityDictionary(dct)
    else:
        raise NotImplementedError