from gpt2_lm import GPT2LM
import pandas as pd
import numpy as np 
import itertools 
import os
import json 

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
            return 0.0

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

    def rerank_dict(self, dict_file, expansions_file):
        with open(dict_file, 'r') as f:
            dict_object = json.load(f)
        with open(expansions_file,'r') as f:
            expansions_object = json.load(f)
        output = {}
        for key, value in dict_object.items():
            std = self.get_key_std(key, value, expansions_object)
            new_value = float(int(value)) + (0.0001 * std)
            output[key] = new_value
        return output

def write_reranked_dicts(file_groups, model_dict, 
    dict_identifier='dict.json', 
    expansions_identifier='expansions.json', 
    prefix='reranked_',
    output_dir='./output'):
    file_list = []
    for path, subpaths, files in os.walk(output_dir):
        for filename in files:
            if filename.endswith(dict_identifier):
                file_list.append(path)

    file_dict = {}
    for key in file_groups:
        file_dict[key] = [ x for x in file_list if key in x]
    
    for key in file_dict.keys():
        rerank = Ranking(model_dict[key])
        for path in file_dict[key]:
            expansions_file = os.path.join(path, expansions_identifier)
            dict_file = os.path.join(path, dict_identifier)
            output = rerank.rerank_dict(dict_file, expansions_file)
            new_dict_file = os.path.join(path, prefix + dict_identifier)
            with open(new_dict_file, 'w+') as f:
                json.dump(output, f)


def main():
    file_groups = [ "small", "medium", "large" ]
    model_dict = {
        "small": "gpt2",
        "medium": "gpt2-medium",
        "large": "gpt2-large"
    }
    write_reranked_dicts(file_groups, model_dict)

if __name__ == '__main__':
    main()