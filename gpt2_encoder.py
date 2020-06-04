import itertools
from nltk.metrics import edit_distance
from collections import defaultdict
import itertools
import json
import torch
import pandas as pd
import numpy as np

from transformers import (
    GPT2Model,
    GPT2Tokenizer,
    HfArgumentParser
)

from configuration_classes import (
   ModelArguments, 
   DataEvaluationArguments,
   EncoderArguments, 
   CNNArguments,
)

class GPT2Encoder(object):

    def __init__(self, 
        model_name_or_path, 
        expansions_json_file, 
        report_json_file, 
        dictionary_file, 
        device='cuda'):

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.expansions_json_file = expansions_json_file
        self.expansions_json_list = []
        self.report_json_file = report_json_file
        self.report_json_list = []
        self.device = device
        self.generations_df = None
        self.pairs_df = None
        with open(dictionary_file,'r') as f:
            self.dictionary = json.load(f)
        with open(self.expansions_json_file,'r') as f:
            self.expansions_json_list = json.load(f)
        with open(self.report_json_file,'r') as f:
            self.report_json_list = json.load(f)

    def compile_generations_df(self):

        expansions_json_list = self.expansions_json_list
        report_json_list = self.report_json_list
        
        expansions = itertools.chain(*expansions_json_list)
        expansions_dict = defaultdict(list)
        for item in expansions:
            expansions_dict[item['hypothesis']] += [item['generation']]
        expansions = [{'hypothesis': k, 'generations': v} for k,v in expansions_dict.items()]
        expansions = pd.DataFrame(expansions)

        report = pd.DataFrame(report_json_list)
        input_df = expansions.merge(report, on='hypothesis', how='left')
        input_df = input_df.drop(['recall', 'precision', 'F1'], axis=1)
        input_df['labels'] = input_df['hypothesis'].combine(input_df['gold'], lambda x,y: 1 if x == y else 0)
        dictionary_df = [ {'hypothesis': k, 'gpt2_score': v} for k,v in self.dictionary.items() ]
        dictionary_df = pd.DataFrame(dictionary_df)
        input_df = input_df.merge(dictionary_df, how='left', on='hypothesis')

        input_df['cnn_score'] = 0.0

        input_df = input_df.sort_values(by='hypothesis')

        input_df['generations'] = input_df['generations'].apply(lambda x: ' '.join(x))
        input_df['generations'] = input_df['generations'].str.replace("\n"," ")

        input_df['characters'] = input_df['hypothesis'].str.replace(" ","")
        input_df = input_df.sort_values(by=['characters', 'gold']).fillna(method='ffill')
        input_df['distance_score'] = input_df['hypothesis'].combine(input_df['gold'], edit_distance)

        input_df = input_df.sort_values(by=['characters', 'hypothesis'])
        self.generations_df = input_df

    def compile_pairs_df(self):
        pairs = self.generations_df[['hypothesis', 'characters', 'gpt2_score', 'cnn_score', 'distance_score']]

        combinations_1 = pairs\
            .groupby('characters')['hypothesis']\
            .apply(lambda x: pd.DataFrame(list(itertools.combinations(x,2))))\
            .reset_index(level=1, drop=True)\
            .reset_index()[[0,1]]

        combinations_2 = combinations_1[[1,0]]
        combinations_2 = combinations_2.rename(columns={
            1: 0,
            0: 1
        })
        combinations = pd.concat([combinations_1, combinations_2])
        combinations['characters'] = combinations[0].str.replace(" ","")
        combinations = combinations.sort_values('characters')
        combinations = combinations.drop('characters', axis=1)

        pairs = pairs.merge(combinations, left_on='hypothesis', right_on=0, how='left')
        pairs = pairs.drop('hypothesis', axis=1)
        pairs = pairs.drop('characters', axis=1)
        pairs = pairs.rename(columns={
            "gpt2_score" : "left_gpt2_score",
            "cnn_score": "left_cnn_score",
            "distance_score": "left_distance_score",
            0: "left_hypothesis",
            1: "right_hypothesis"
        })
        pairs = pairs.merge(pairs, left_on='right_hypothesis', right_on='left_hypothesis', how='left')
        pairs = pairs.drop("left_hypothesis_y", axis=1)
        pairs = pairs.drop("right_hypothesis_y", axis=1)

        pairs = pairs.rename(columns={
            "left_gpt2_score_x": "left_gpt2_score",
            "left_cnn_score_x": "left_cnn_score",
            "left_distance_score_x": "left_distance_score",
            "left_hypothesis_x": "left_hypothesis",
            "right_hypothesis_x": "right_hypothesis",
            "left_gpt2_score_y": "right_gpt2_score",
            "left_cnn_score_y": "right_cnn_score",
            "left_distance_score_y": "right_distance_score"
        })

        pairs["distance_score"] = pairs["left_distance_score"]\
            .combine(pairs["right_distance_score"], lambda x, y: x - y)
        cols = pairs.columns.values.tolist()
        move_first = [
            'left_gpt2_score', 
            'left_cnn_score', 
            'right_gpt2_score', 
            'right_cnn_score', 
            'distance_score']
        cols = [ x for x in cols if x not in move_first ]
        pairs = pairs[[*move_first, *cols]]
        self.pairs_df = pairs

    def _get_generations_and_labels(self):
        generations_list = self.generations_df.generations.values.tolist()
        labels_list = self.generations_df.labels.values.tolist()
        return generations_list, labels_list

    def get_input_ids_and_labels(self, trim=True, max_length=None, max_model_length=1024):
        generations_list, labels_list = self._get_generations_and_labels()
        input_ids = [ self.tokenizer.encode(x) for x in generations_list ]
        input_ids = [ torch.tensor(x) for x in input_ids ]
        if not max_length:
            max_length = min([x.shape[0] for x in input_ids])
        if max_length > max_model_length:
            max_length = max_model_length
        if trim:
            input_ids = [ x[0:max_length] for x in input_ids ]
        labels = torch.tensor(labels_list)
        return input_ids, labels

    def update_cnn_values(self, cnn_values):
        self.generations_df['cnn_score'] = pd.Series(cnn_values)

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataEvaluationArguments, CNNArguments, EncoderArguments))
    model_args, data_args, cnn_args, encoder_args = parser.parse_args_into_dataclasses()

    gpt2 = GPT2Encoder(
        model_args.model_name_or_path, 
        data_args.expansions_file, 
        data_args.report_file, 
        data_args.dict_file)

    if encoder_args.compile_generations:
        gpt2.compile_generations_df()
        gpt2.generations_df.to_csv(encoder_args.generations_file)
    if encoder_args.compile_pairs:
        gpt2.compile_pairs_df()
        gpt2.pairs_df.to_csv(encoder_args.pairs_file)
    