import itertools
from nltk.metrics import edit_distance
from collections import defaultdict
import itertools
import json
import torch
import pandas as pd
import numpy as np
# import dask.dataframe as dd
# import dask
# import h5py
import logging
import logging.config
from scipy.stats import skew, kurtosis

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
        self.logger = logging.getLogger(__name__)

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
        input_df = input_df.explode('generations').reset_index(drop=True)

        input_df['characters'] = input_df['hypothesis'].str.replace(" ","")
        input_df = input_df.sort_values(by=['characters', 'gold']).fillna(method='ffill')

        input_df = input_df.astype({
            "hypothesis": "str",
            "generations": "str",
            "gold": "str",
            "labels": "int64",
            "gpt2_score": "float64",
            "cnn_score": "float64",
            "characters": "str"
        })

        input_df['distance_score'] = input_df['hypothesis'].combine(input_df['gold'], edit_distance)

        input_df = input_df.sort_values(by=['characters', 'hypothesis'])
        self.generations_df = input_df

    def _generate_combinations(self, pairs):
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
        combinations = combinations.rename(columns={
            0: 'hypothesis',
            1: 'hypothesis_2'
        })
        return combinations

    def _merge_pairs_and_combinations(self, pairs, combinations):
        pairs = pairs.merge(combinations, on='hypothesis', how='left')

        pairs_mirror = pairs.copy()
        pairs_mirror = pairs_mirror.drop(['hypothesis_2','characters'], axis=1)

        pairs = pairs.merge(pairs_mirror, 
            left_on='hypothesis_2', 
            right_on='hypothesis', 
            how='left', 
            suffixes=('_1', '_2'))

        pairs = pairs.loc[:,~pairs.columns.duplicated()]
        pairs['distance_score'] = pairs['distance_score_1'] - pairs['distance_score_2']
        return pairs

    def _generate_stats(self, pairs_df):
        stats_df = pairs_df[['hypothesis', 'cnn_score']].groupby('hypothesis').agg([
            np.mean, 
            np.std, 
            np.max, 
            np.min, 
            skew, 
            kurtosis]).reset_index(drop=False)
        stats_df.columns = stats_df.columns.droplevel(-2)
        stats_df = stats_df.rename(columns={stats_df.columns[0]: 'hypothesis'})
        pairs_df = pairs_df.drop_duplicates(subset=[
            'hypothesis', 
            'characters', 
            'gpt2_score', 
            'distance_score'])
        pairs_df = pairs_df.merge(stats_df, on='hypothesis', how='outer')
        pairs_df = pairs_df.drop('cnn_score',axis=1)
        return pairs_df

    def compile_pairs_df(self, keep_df=False):

        pairs = self.generations_df[[
            'hypothesis', 
            'characters', 
            'gpt2_score', 
            'cnn_score', 
            'distance_score']]

        pairs = self._generate_stats(pairs)

        pairs = pairs[[
            'mean',
            'std',
            'amax',
            'amin',
            'skew',
            'kurtosis',
            'gpt2_score',
            'distance_score',
            'hypothesis',
            'characters'
        ]]

        combinations = self._generate_combinations(pairs)

        pairs = self._merge_pairs_and_combinations(pairs, combinations)

        if keep_df:
            self.pairs_df = pairs.copy()
            assert(self.pairs_df.shape == self.pairs_df.dropna().shape)

        pairs = pairs[[
                'mean_1',
                'std_1',
                'amax_1',
                'amin_1',
                'skew_1',
                'kurtosis_1',
                'gpt2_score_1',
                'mean_2',
                'std_2',
                'amax_2',
                'amin_2',
                'skew_2',
                'kurtosis_2',
                'gpt2_score_2',
                'distance_score']]

        pairs_array = np.array(pairs.values.tolist())
        self.logger.debug('pairs_array shape: {0}'.format(pairs_array.shape))
        return pairs_array


    def _get_generations_and_labels(self):
        generations_list = self.generations_df.generations.values.tolist()
        labels_list = self.generations_df.labels.values.tolist()
        return generations_list, labels_list

    def get_input_ids_and_labels(self, trim=True, min_length=None, max_length=None, max_model_length=1024):
        generations_list, labels_list = self._get_generations_and_labels()
        input_ids = [ self.tokenizer.encode(x) for x in generations_list ]
        input_ids = [ torch.tensor(x) for x in input_ids ]
        
        if trim:
            if not min_length:
                min_length = int(np.average([len(x) for x in input_ids]))

            input_ids = [ x for x in input_ids if len(x) >= min_length ]

            if not max_length:
                max_length = min([x.shape[0] for x in input_ids])

            if max_length > max_model_length:
                max_length = max_model_length

            input_ids = [ x[0:max_length] for x in input_ids ]
        
        labels = torch.tensor(labels_list)
        
        return input_ids, labels

    def update_cnn_values(self, cnn_values):
        self.generations_df['cnn_score'] = pd.Series(cnn_values)
        assert(self.generations_df.dropna().shape == self.generations_df.shape)

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
    