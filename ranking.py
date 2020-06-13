from gpt2_lm import GPT2LM
import pandas as pd
import numpy as np 
import itertools 
import os
import json 

from transformers import (
    HfArgumentParser
)

from configuration_classes import (
    parameters_to_command,
    parameters_to_string,
    ModelArguments, 
    DataEvaluationArguments, 
    BeamsearchArguments, 
    RankingArguments,
    BeamsearchManagerArguments
)


class Ranking(object):

    def __init__(self, model_name_or_path):
        self.gpt2 = GPT2LM(model_name_or_path)

    def calculate_score(self, a, b):
        gpt2_input = a + b
        score = self.gpt2.get_probs([gpt2_input])
        return score[0]

    def get_key_std(self, key, value, expansions):
        expansions = list(itertools.chain(*expansions))
        expansions = pd.DataFrame(expansions)
        expansions = expansions[expansions['hypothesis']==key]
        
        if expansions.dropna().empty:
            return None

        combinations_1 = expansions\
        .groupby('hypothesis')['generation']\
        .apply(lambda x: pd.DataFrame(list(itertools.combinations(x,2))))\
        .reset_index(drop=False)\
        .reset_index()[['hypothesis', 0, 1]]
        combinations_1 = combinations_1.rename(columns={
            0: "g1",
            1: "g2"
        })
        assert(combinations_1.shape == combinations_1.dropna().shape)
        combinations_1['score'] = combinations_1['g1'].combine(combinations_1['g2'], self.calculate_score)
        std = np.std(combinations_1['score'].astype(float).values.tolist())
        return std

    def rerank(self, decimal, fractional, method='std', order=0.0001):
        if method == 'std':
            decimal = float(int(decimal))
            fractional = order * fractional
            output = decimal + fractional
            return output 

    def get_allowed_items(self, dict_object, span=4):
        df = pd.DataFrame([ {'hypothesis': k, 'perplexity': v} for k,v in dict_object.items() ])
        df['characters'] = df['hypothesis'].str.replace(" ", "")
        df = df.sort_values('perplexity',ascending=True).groupby('characters', sort=False).head(span)
        allowed_items = df['hypothesis'].values.tolist()
        return allowed_items

    def rerank_dict(self, dict_file, expansions_file):
        with open(dict_file, 'r') as f:
            dict_object = json.load(f)
        with open(expansions_file,'r') as f:
            expansions_object = json.load(f)
        output = {}
        allowed_items = self.get_allowed_items(dict_object)
        for key, value in dict_object.items():
            if key in allowed_items:
                std = self.get_key_std(key, value, expansions_object)
                if std:
                    new_value = self.rerank(value, std)
                    output[key] = new_value
                else:
                    output[key] = value
            else:
                output[key] = value
        return output

def write_dict_file(filename, output):
    with open(filename, 'w+') as f:
        json.dump(output, f)

def write_reranked_dicts(model_name_or_path, dict_file, expansions_file, prefix='rerank_'):
    dict_filename = os.path.split(dict_file)[1]
    dict_path = os.path.split(dict_file)[0]
    new_dict_file = os.path.join(dict_path, prefix + dict_filename)
    if not os.path.isfile(new_dict_file):
        rerank = Ranking(model_name_or_path)
        output = rerank.rerank_dict(dict_file, expansions_file)
        write_dict_file(new_dict_file, output)

def main():
    parser = HfArgumentParser((ModelArguments, DataEvaluationArguments, BeamsearchArguments, RankingArguments, BeamsearchManagerArguments))
    model_args, data_args, beam_args, ranking_args, manager_args = parser.parse_args_into_dataclasses()
    write_reranked_dicts(model_args.model_name_or_path, data_args.dict_file, data_args.expansions_file)

if __name__ == '__main__':
    main()