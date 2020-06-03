import copy
import itertools
import multiprocessing
import re
import sys
import numpy as np
import torch
import pandas as pd
import logging
import json
from timeit import default_timer as timer
from collections import defaultdict
from pathos.multiprocessing import ProcessingPool as Pool
from transformers import BertForMaskedLM, BertTokenizer
from concurrent.futures import ThreadPoolExecutor
from gpt2_lm import GPT2LM
from bert_lm import BertLM
from reader import DatasetReader

from configuration_classes import (
    ModelArguments,
    DataEvaluationArguments,
    BeamsearchArguments,
    RankingArguments
)

from transformers import (
    HfArgumentParser
)

from metrics import (
    calculate_recall,
    calculate_precision,
    calculate_f1
)

logger = logging.getLogger(__name__)

class ModelLM(object):

    def __init__(self, model_name_or_path=None, model_type=None, device=None, gpu_batch_size=None):
        if model_type == 'gpt2':
            self.model = GPT2LM(model_name_or_path, device=device, gpu_batch_size=gpu_batch_size)
        elif model_type == 'bert':
            self.model = BertLM(model_name_or_path, device=device, gpu_batch_size=gpu_batch_size)
        else:
            raise NotImplementedError

class Beamsearch(ModelLM):

    def __init__(self, dataset=None, model_name_or_path=None, model_type=None, device='cuda', gpu_batch_size=1, chunks=1):
        self.dataset = dataset
        self.tree = []
        self.ids = []
        self.probs = []
        self.prob_dict = {}
        super().__init__(model_name_or_path=model_name_or_path, model_type=model_type, device=device, gpu_batch_size=gpu_batch_size)
        self.gpu_batch_size = gpu_batch_size

    def next_step(self, list_of_candidates):
        output = []
        for candidate_string in list_of_candidates:
            candidates = [ candidate_string[:pos] + ' ' + candidate_string[pos:] if pos else candidate_string for pos in range(len(candidate_string)) ]
            candidates = list(filter(lambda x: not re.findall(".*?(?=\s{2})",x), candidates))
            output.extend(candidates)
        return output

    def update_probabilities(self):
        for item in self.tree:
            current_batch = []
            for word in item:
                if word in self.prob_dict:
                    continue
                else:
                    current_batch.append(word)
            if current_batch:
                current_batch_probs = self.model.get_probs(current_batch)
            for idx, word in enumerate(current_batch):
                self.prob_dict[word] = current_batch_probs[idx]

    def reshape_tree(self, measure):
        self.tree = [ self.tree[x:x+measure] for x in range(0, len(self.tree), measure) ]

    def flatten_list(self, list_):
        return [ item for sublist in list_ for item in sublist ]

    def trim_tree(self, topk):
        output = []
        probs = [ self.prob_dict[x] for x in self.tree ]
        candidates = [ { 'hypothesis': item, 'characters': item.replace(" ", ""), 'score': probs[idx] } for idx, item in enumerate(self.tree) ]
        for key, group in itertools.groupby(candidates, key=lambda x:x['characters']):
            sorted_group = sorted(list(group), key=lambda x: x['score'])
            trimmed_group = sorted_group[0:topk]
            trimmed_group = [x['hypothesis'] for x in trimmed_group]
            output.extend(trimmed_group)
        return output

    def convert_dataset(self):
        guesses = []
        candidates = [ { 'hypothesis': k, 'score': v, 'characters': k.replace(" ", "") } for k,v in self.prob_dict.items() ]
        candidates = sorted(candidates, key=lambda x: x['characters'])
        for key, group in itertools.groupby(candidates, key=lambda x:x['characters']):
            best_guess = sorted(list(group), key=lambda x: x['score'])[0]
            guesses.append(best_guess)
        segmentation_dict = {}
        for idx, item in enumerate(guesses):
            segmentation_dict[item['characters']] = item['hypothesis']

        output = []
        for idx, item in enumerate(self.dataset):
            if item:
                output.append(segmentation_dict[item])
        
        return output

    def run(self, topk, steps):
        self.tree = copy.deepcopy(self.dataset)
        for i in range(steps):

            self.tree = self.next_step(self.tree)

            self.reshape_tree(self.gpu_batch_size)
            self.update_probabilities()
            self.tree = self.flatten_list(self.tree)

            self.tree = self.trim_tree(topk)

def evaluate_word_segmentation_model(segmented_data, original_data, save_report=True, report_file='report.json'):
    pairs = list(zip(segmented_data, original_data))
    metrics = {
        "recall": 0.,
        "precision": 0.,
        "F1": 0.
    }
    report = []
    for idx, item in enumerate(pairs):
        recall = calculate_recall(*item)
        precision = calculate_precision(*item)
        F1 = calculate_f1(*item)
        metrics['recall'] += recall
        metrics['precision'] += precision
        metrics['F1'] += F1
        report_row = {
            "gold": item[1],
            "hypothesis": item[0],
            "recall": recall,
            "precision": precision,
            "F1": F1
        }
        report.append(report_row)
    
    for k,v in metrics.items():
        metrics[k] = v / len(pairs)
    
    if save_report:
        with open(report_file,'w+') as f:
            json.dump(report, f)

    return metrics

def gather_n_candidates(prob_dict, segmented_data, n=2):
    character_count = 0
    guesses = []
    candidates = [ { 'hypothesis': k, 'score': v, 'characters': k.replace(" ", "") } \
        for k,v in prob_dict.items() ]
    candidates = sorted(candidates, key=lambda x: x['characters'])
    for key, group in itertools.groupby(candidates, key=lambda x:x['characters']):
        best_guess = sorted(list(group), key=lambda x: x['score'])[0:n]
        best_guess = [ x['hypothesis'] for x in best_guess ]
        character_count += sum([len(x) for x in best_guess])
        guesses.append(best_guess)
    
    return guesses, character_count

def main():
    parser = HfArgumentParser((ModelArguments, DataEvaluationArguments, BeamsearchArguments, RankingArguments))
    model_args, data_args, beam_args, ranking_args = parser.parse_args_into_dataclasses()
    reader = DatasetReader(data_args.eval_data_file, data_args.eval_dataset_format)
    reader.read()
    if data_args.n_chunks != -1 and data_args.index != -1:
        reader.trim(data_args.index, data_args.n_chunks)
    
    beamsearch = Beamsearch(
        dataset=reader.dataset, 
        model_name_or_path=model_args.model_name_or_path, 
        model_type=model_args.model_type,
        gpu_batch_size=model_args.gpu_batch_size)
    
    start_time = timer()
    beamsearch.run(beam_args.topk, beam_args.steps)
    end_time = timer()

    elapsed_time = end_time - start_time
    print('elapsed time: ', elapsed_time)
    segmented_data = beamsearch.convert_dataset()
    result = evaluate_word_segmentation_model(
                    segmented_data, 
                    reader.test, 
                    save_report=True, 
                    report_file=data_args.report_file)
    print(result)
    print('characters / second: ', reader.character_count / elapsed_time)

    with open(data_args.dict_file, 'w+') as f:
        json.dump(beamsearch.prob_dict, f)
    
    if model_args.model_type == 'gpt2':
        groups, groups_character_count = \
            gather_n_candidates(beamsearch.prob_dict, segmented_data, n=ranking_args.topn)
        start_time = timer()
        beamsearch.model.gpu_expansion_batch_size = ranking_args.gpu_expansion_batch_size
        for group in groups:
            beamsearch.model.generate_expansions(group)
        new_segmented_data = beamsearch.model.expansions
        end_time = timer()
        elapsed_time = end_time - start_time

        print('elapsed time: ', elapsed_time)    
        print('characters / second: ', groups_character_count / elapsed_time)

        with open(data_args.expansions_file, 'w+') as f:
            json.dump(new_segmented_data, f)

if __name__ == '__main__':
    main()