import pandas as pd
from dataclasses import dataclass

@dataclass
class Node:
    hypothesis: str
    characters: str
    score: float

@dataclass
class ProbabilityDictionary(object):
    dictionary: dict

    def get_top_k(
        self,
        k=2,
        characters_field='characters',
        segmentation_field='segmentation',
        score_field='score'
    ):
        df = self.to_dataframe(
            characters_field=characters_field,
            segmentation_field=segmentation_field,
            score_field=score_field
        )
        df = df.groupby(
            by=characters_field
        ).head(k)

        keys = df[segmentation_field].values
        values = df[score_field].values
        output = {
            k:v for k,v in list(zip(keys, values))
        }
        return output

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