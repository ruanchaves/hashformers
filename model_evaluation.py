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

def record_to_ranking(record,
    left_hypothesis_field='hypothesis_1',
    right_hypothesis_field='hypothesis_2',
    left_gpt2_score_field='gpt2_score_1',
    right_gpt2_score_field='gpt2_score_2'):
    if record[left_gpt2_score_field] < record[right_gpt2_score_field]:
        gpt2_ranking = {
            'good': record[left_hypothesis_field], 
            'bad' : record[right_hypothesis_field] 
        }
    else:
        gpt2_ranking = {
            'bad': record[left_hypothesis_field], 
            'good' : record[right_hypothesis_field]             
        }
        
    if record['predicted_distance_score'] <= 0.0:
        mlp_ranking = {
            'good': record[left_hypothesis_field], 
            'bad' : record[right_hypothesis_field] 
        }        
    else:
        mlp_ranking = {
            'bad': record[left_hypothesis_field], 
            'good' : record[right_hypothesis_field]             
        }
        
    if record['distance_score'] <= 0.0:
        gold_ranking = {
            'good': record[left_hypothesis_field], 
            'bad' : record[right_hypothesis_field] 
        }
    else:
        gold_ranking = {
            'bad': record[left_hypothesis_field], 
            'good' : record[right_hypothesis_field]             
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

def calculate_metrics(df, logger=None):
    df['characters'] = df['hypothesis_1'].str.replace(" ","")
    df_items = df.to_dict('records')
    characters_dict = defaultdict(list)
                    
    for item in df_items:
        characters_dict[item['characters']].append(item)

    for key, value in characters_dict.items():
        characters_dict[key] = [ record_to_ranking(x) for x in value ]

    logger.debug('characters_dict length: {0}'.format(len(characters_dict)))
        
    mlp_dict = defaultdict(list)
    gpt2_dict = defaultdict(list)
    gold_dict = defaultdict(list)

    for key, value in characters_dict.items():
        gold_dict[key] = sort_pairwise(value, model_key="gold_ranking")

    for key, value in characters_dict.items():
        mlp_dict[key] = sort_pairwise(value, model_key="mlp_ranking")

    for key, value in characters_dict.items():
        gpt2_dict[key] = sort_pairwise(value, model_key="gpt2_ranking")

    logger.debug("\n gold_dict length: {0} \n mlp_dict length: {1} \n gpt2_dict length:{2}"\
        .format(len(gold_dict), len(mlp_dict), len(gpt2_dict)))

    unit_size = len(list(gold_dict.values())[0])

    hypothesis_to_id = { hypothesis: idx  \
        for idx, hypothesis in enumerate(list(itertools.chain(*gold_dict.values()))) }

    gold_dict_ids = [ hypothesis_to_id[hypothesis] \
        for idx, hypothesis in enumerate(list(itertools.chain(*gold_dict.values()))) ]

    gpt2_dict_ids = [ hypothesis_to_id[hypothesis] \
        for idx, hypothesis in enumerate(list(itertools.chain(*gpt2_dict.values()))) ]

    mlp_dict_ids = [ hypothesis_to_id[hypothesis] \
        for idx, hypothesis in enumerate(list(itertools.chain(*mlp_dict.values()))) ]

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

    evaluation_df['gpt2_hypothesis'] = evaluation_df['gpt2_hypothesis'].astype(str)
    evaluation_df['gold_hypothesis'] = evaluation_df['gold_hypothesis'].astype(str)
    evaluation_df['mlp_hypothesis'] = evaluation_df['mlp_hypothesis'].astype(str)

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

    logging.config.fileConfig('logging.conf', \
        defaults={'logfilename': data_args.logfile})
    logger = logging.getLogger(__file__)

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

    batch_size = 1
    input_ids = [torch.stack(input_ids[i:i+batch_size]) for i in range(0, len(input_ids), batch_size)]

    cnn_probs = [ x.item() for x in \
        cnn_model.predict(input_ids) ]

    gpt2_encoder.update_cnn_values(cnn_probs)

    pairs_array = gpt2_encoder.compile_pairs_df(keep_df=True)
    features = [ x[0:-1] for x in pairs_array ]

    features = torch.tensor(features, 
        dtype=torch.float, 
        device=mlp_args.mlp_device)
    
    results = []
    features = torch.split(features, 1)
    features = torch.stack(features)
    for result_tensor in mlp_model.predict(features):
        pair_prediction = result_tensor.item()
        results.append(pair_prediction)

    gpt2_encoder.pairs_df['predicted_distance_score'] = pd.Series(results)
    results = calculate_metrics(gpt2_encoder.pairs_df, logger=logger)
    logger.info('\n{0}\n{1}'.format(
        results['kendall'].to_string(index=False), 
        results['other'].to_string()))

if __name__ == '__main__':
    main()