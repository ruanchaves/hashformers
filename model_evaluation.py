import torch

import itertools
from collections import defaultdict
import numpy as np
import scipy
from metrics import calculate_recall, calculate_f1, calculate_precision

from transformers import (
    HfArgumentParser
)

from configuration_classes import (
   parameters_to_string,
   ModelArguments, 
   DataEvaluationArguments,
   EncoderArguments, 
   CNNArguments,
   MLPArguments,
   ModelEvaluationArguments
)

from cnn_model import CNNModel
from mlp_model import MLPModel
from gpt2_encoder import GPT2Encoder

import pandas as pd

import logging

class WordSegmentationHypothesis(object):
    
    def __init__(self, hypothesis, pair_list):
        self.hypothesis = hypothesis
        self.pair_list = pair_list
    
    def better_than_other(self, other):
        for item in self.pair_list:
            if item['good'] == self.hypothesis and item['bad'] == other.hypothesis:
                return True
        return False
    
    def __lt__(self, other):
        # x < y
        compare = self.better_than_other(other)
        if compare:
            return False
        else:
            return True
        
    def __le__(self, other):
        # x <= y
        compare = self.better_than_other(other)
        if compare:
            return False
        else:
            return True        
        
    def __eq__(self, other):
        # x == y
        return False
        
    def __ne__(self, other):
        # x != y
        return True
        
    def __gt__(self, other):
        # x > y
        compare = self.better_than_other(other)
        if compare:
            return True
        else:
            return False
        
    def __ge__(self, other):
        # x >= y
        compare = self.better_than_other(other)
        if compare:
            return True
        else:
            return False      

def record_to_ranking(record):
    if record['left_gpt2_score'] < record['right_gpt2_score']:
        gpt2_ranking = {
            'good': record['left_hypothesis'], 
            'bad' : record['right_hypothesis'] 
        }
    else:
        gpt2_ranking = {
            'bad': record['left_hypothesis'], 
            'good' : record['right_hypothesis']             
        }
        
    if record['predicted_distance_score'] <= 0.0:
        mlp_ranking = {
            'good': record['left_hypothesis'], 
            'bad' : record['right_hypothesis'] 
        }        
    else:
        mlp_ranking = {
            'bad': record['left_hypothesis'], 
            'good' : record['right_hypothesis']             
        }
        
    if record['distance_score'] <= 0.0:
        gold_ranking = {
            'good': record['left_hypothesis'], 
            'bad' : record['right_hypothesis'] 
        }
    else:
        gold_ranking = {
            'bad': record['left_hypothesis'], 
            'good' : record['right_hypothesis']             
        }
        
    ranking = {
        "gpt2_ranking": gpt2_ranking,
        "mlp_ranking": mlp_ranking,
        "gold_ranking": gold_ranking
    }
    return ranking 
        

def sort_pairwise(pairs, model_key='gpt2_ranking'):
    hypothesis_set = set()
    for pair_item in pairs:
        for status in ['good', 'bad']:
            hypothesis_set.update([pair_item[model_key][status]])
    
    pair_list = []
    for pair_item in pairs:
        pair_list.append(pair_item[model_key])
        
    hypothesis_list = list(hypothesis_set)
    for idx, item in enumerate(hypothesis_list):
        hypothesis_list[idx] = WordSegmentationHypothesis(item, pair_list)
        
    sorted_hypothesis_list = sorted(hypothesis_list, reverse=True)
    
    for idx, item in enumerate(sorted_hypothesis_list):
        sorted_hypothesis_list[idx] = item.hypothesis
        
    return sorted_hypothesis_list

def calculate_kendall(gold_dict_ids, model_dict_ids):
    compare = list(zip(gold_dict_ids, model_dict_ids))
    correlation_array = [ scipy.stats.kendalltau(x,y).correlation for x,y in compare ]
    p_value_array = [ scipy.stats.kendalltau(x,y).pvalue for x,y in compare ]
    average_tau = np.mean(correlation_array)
    std_tau = np.std(correlation_array)
    average_p_value = np.mean(p_value_array)
    std_p_value = np.std(p_value_array)
    result = {
        "average_tau": average_tau,
        "std_tau": std_tau,
        "average_pvalue": average_p_value,
        "std_pvalue": std_p_value
    }
    return result

def calculate_metrics(df):
    df['characters'] = df['left_hypothesis'].str.replace(" ","")
    df_items = df.to_dict('records')
    characters_dict = defaultdict(list)
                    
    for item in df_items:
        characters_dict[item['characters']].append(item)

    for key, value in characters_dict.items():
        characters_dict[key] = [ record_to_ranking(x) for x in value ]
        
    mlp_dict = defaultdict(list)
    gpt2_dict = defaultdict(list)
    gold_dict = defaultdict(list)

    for key, value in characters_dict.items():
        gold_dict[key] = sort_pairwise(value, model_key="gold_ranking")

    for key, value in characters_dict.items():
        mlp_dict[key] = sort_pairwise(value, model_key="mlp_ranking")

    for key, value in characters_dict.items():
        gpt2_dict[key] = sort_pairwise(value, model_key="gpt2_ranking")

    unit_size = len(list(gold_dict.values())[0])

    hypothesis_to_id = { hypothesis: idx  for idx, hypothesis in enumerate(list(itertools.chain(*gold_dict.values()))) }

    gold_dict_ids = [ hypothesis_to_id[hypothesis] for idx, hypothesis in enumerate(list(itertools.chain(*gold_dict.values()))) ]

    gpt2_dict_ids = [ hypothesis_to_id[hypothesis] for idx, hypothesis in enumerate(list(itertools.chain(*gpt2_dict.values()))) ]

    mlp_dict_ids = [ hypothesis_to_id[hypothesis] for idx, hypothesis in enumerate(list(itertools.chain(*mlp_dict.values()))) ]

    gold_dict_ids = [gold_dict_ids[x:x+unit_size] for x in range(0, len(gold_dict_ids), unit_size)]
    gpt2_dict_ids = [gpt2_dict_ids[x:x+unit_size] for x in range(0, len(gpt2_dict_ids), unit_size)]
    mlp_dict_ids = [mlp_dict_ids[x:x+unit_size] for x in range(0, len(mlp_dict_ids), unit_size)]

    gold_kendall = calculate_kendall(gold_dict_ids, gold_dict_ids)
    gold_kendall['model'] = 'gold'
    gpt2_kendall = calculate_kendall(gold_dict_ids, gpt2_dict_ids)
    gpt2_kendall['model'] = 'gpt2'
    mlp_kendall = calculate_kendall(gold_dict_ids, mlp_dict_ids)
    mlp_kendall['model'] = 'mlp'

    kendall_df = pd.DataFrame([gpt2_kendall, mlp_kendall, gold_kendall])
    kendall_df_columns = (['model'] + kendall_df.columns.values.tolist())[0:-1]
    kendall_df = kendall_df[[*kendall_df_columns]]

    better_gpt2_dict = { k: v[0] for k,v in gpt2_dict.items() }
    better_mlp_dict = { k: v[0] for k,v in mlp_dict.items() } 
    better_gold_dict = { k: v[0] for k,v in gold_dict.items() }

    better_gpt2_dict_df = [ {'characters': k, 'gpt2_hypothesis': v} for k,v in better_gpt2_dict.items() ]
    better_mlp_dict_df = [ {'characters': k, 'mlp_hypothesis': v} for k,v in better_mlp_dict.items() ]
    better_gold_dict_df = [ {'characters': k, 'gold_hypothesis': v} for k,v in better_gold_dict.items() ]

    better_gpt2_dict_df = pd.DataFrame(better_gpt2_dict_df)
    better_mlp_dict_df = pd.DataFrame(better_mlp_dict_df)
    better_gold_dict_df = pd.DataFrame(better_gold_dict_df)

    evaluation_df = better_gpt2_dict_df.merge(better_mlp_dict_df, how='outer', on='characters')
    evaluation_df = evaluation_df.merge(better_gold_dict_df, how='outer', on='characters')
    assert(evaluation_df.shape[0] == evaluation_df.dropna().shape[0])

    evaluation_df['gpt2_recall'] = evaluation_df['gpt2_hypothesis'].combine(evaluation_df['gold_hypothesis'], calculate_recall)
    evaluation_df['gpt2_f1'] = evaluation_df['gpt2_hypothesis'].combine(evaluation_df['gold_hypothesis'], calculate_f1)
    evaluation_df['gpt2_precision'] = evaluation_df['gpt2_hypothesis'].combine(evaluation_df['gold_hypothesis'], calculate_precision)

    evaluation_df['mlp_recall'] = evaluation_df['mlp_hypothesis'].combine(evaluation_df['gold_hypothesis'], calculate_recall)
    evaluation_df['mlp_f1'] = evaluation_df['mlp_hypothesis'].combine(evaluation_df['gold_hypothesis'], calculate_f1)
    evaluation_df['mlp_precision'] = evaluation_df['mlp_hypothesis'].combine(evaluation_df['gold_hypothesis'], calculate_precision)

    results_df = evaluation_df[['gpt2_recall', 'gpt2_f1', 'gpt2_precision', 'mlp_recall', 'mlp_f1', 'mlp_precision']].agg([np.mean, np.std])

    final_results = {
        "kendall": kendall_df,
        "other": results_df
    }

    return final_results

def main():
    parser = HfArgumentParser((ModelArguments, 
        DataEvaluationArguments, 
        CNNArguments, 
        EncoderArguments, 
        MLPArguments, 
        ModelEvaluationArguments))
    model_args, data_args, cnn_args, encoder_args, mlp_args, model_evaluation_args = parser.parse_args_into_dataclasses()


    # create logger with __file__
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(data_args.logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('\n' + parameters_to_string(model_args, data_args, cnn_args, encoder_args, mlp_args, model_evaluation_args))
 
    mlp_model = MLPModel(model_args, data_args, cnn_args, mlp_args)
    mlp_model.from_pretrained()

    cnn_model = CNNModel(model_args, data_args, cnn_args)
    cnn_model.from_pretrained()

    gpt2_encoder = GPT2Encoder(
    model_args.model_name_or_path, 
    data_args.expansions_file, 
    data_args.report_file, 
    data_args.dict_file)

    gpt2_encoder.compile_generations_df()
    input_ids, _ = gpt2_encoder.get_input_ids_and_labels()

    cnn_probs = [ x.item() for x in \
        cnn_model.predict(input_ids) ]

    gpt2_encoder.update_cnn_values(cnn_probs)
    
    gpt2_encoder.compile_pairs_df()

    features_df = gpt2_encoder.pairs_df[[
        'left_gpt2_score',
        'left_cnn_score',
        'right_gpt2_score',
        'right_cnn_score',
        'distance_score'
    ]]

    features_df_values = features_df.values.tolist()
    features = [ x[0:4] for x in features_df_values ]
    features = torch.tensor(features, 
        dtype=torch.float, 
        device=mlp_args.mlp_device)
    
    results = []
    for result_tensor in mlp_model.predict(features):
        pair_prediction = result_tensor.item()
        results.append(pair_prediction)

    gpt2_encoder.pairs_df['predicted_distance_score'] = pd.Series(results)
    results = calculate_metrics(gpt2_encoder.pairs_df)
    logger.info('\n{0}\n{1}'.format(
        results['kendall'].to_string(index=False), 
        results['other'].to_string()))

if __name__ == '__main__':
    main()