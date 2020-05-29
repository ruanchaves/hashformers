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
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from pathos.multiprocessing import ProcessingPool as Pool
from transformers import BertForMaskedLM, BertTokenizer

from lm_scorer.models.auto import AutoLMScorer as LMScorer
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    HfArgumentParser,
    GPT2LMHeadModel, 
    GPT2Tokenizer
)

from metrics import (
    calculate_recall,
    calculate_precision,
    calculate_f1
)


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint."
        },
    )

    model_type: str = field(
        default='gpt2',
                metadata={
            "help": "Either 'bert' or 'gpt2'."
        }
    )
    
    gpu_batch_size: int = field(
        default=2,
        metadata={"help": "GPU batch size for candidate evaluation."},
    )

@dataclass
class DataEvaluationArguments:
    """
    Arguments pertaining to what data we are going to input our model for evaluation.
    """

    eval_data_file: str = field(
        default=None,
        metadata={"help": "The evaluation data file."},
    )

    eval_dataset_format: Optional[str] = field(
        default="default",
        metadata={"help": "The evaluation data format."},
    )

@dataclass
class BeamsearchArguments:

    topk: int = field(
        default=10,
        metadata={"help": "Beam size."},
    )

    steps: int = field(
        default=2,
        metadata={"help": "Tree depth (maximum amount of words - 1)."},
    )

class DatasetReader(object):

    def __init__(self, dataset_file, dataset_format):
        print(dataset_file)
        self.dataset_file = dataset_file
        self.format = dataset_format
        self.dataset = []
        self.test = []
        self.character_count = 0

    def read(self):
        def error_handling():
            self.default()
        getattr(self,self.format,error_handling)()
        self.character_count = sum([len(x) for x in self.dataset])

    def default(self):
        self.BOUN()

    def BOUN(self):
        with open(self.dataset_file,'r') as f:
            data = f.read().split('\n')
        data = [ x.strip() for x in data ]
        self.test = data[::]
        data = [ x.replace(" ","") for x in data ]
        self.dataset = data[::]

    def glushkova(self):
        data = pd.read_csv(self.dataset_file)
        self.dataset = data['hashtag'].astype(str).values.tolist()
        data['test'] = data['hashtag'].combine(data['true_segmentation'], self.labels_to_tokens)
        self.test = data['test'].astype(str).values.tolist()

    def labels_to_tokens(self, tokens, labels):
        tokens = [ x for x in tokens ] 
        labels = [ None if x == 0 else ' ' for x in labels ]
        new_tokens = list(itertools.chain(*zip(tokens, labels)))
        new_tokens = list(filter(lambda x: x is not None, new_tokens))
        new_tokens = "".join(new_tokens)
        return new_tokens


class BertLM(object):

    def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.model = BertForMaskedLM.from_pretrained(model_name_or_path)
        self.device = device
        self.model.to(device, non_blocking=True)
        self.model.eval()

    def batch_encode(self,list_of_strings):
        batch_encoding = [ self.tokenizer.encode_plus(item) for item in list_of_strings ]

        input_ids = np.array([x['input_ids'] for x in batch_encoding ])
        attention_mask = np.array([x['attention_mask'] for x in batch_encoding ])

        input_ids = self.insert_padding(input_ids, 0)
        attention_mask = self.insert_padding(attention_mask, 0)
        
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        return input_ids, attention_mask
    
    def calculate_loss(self,input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.model(input_ids=input_ids, masked_lm_labels=input_ids, attention_mask=attention_mask, encoder_attention_mask=attention_mask)
        for idx, item in enumerate(attention_mask):
            current_mask = item.view(1,-1).transpose(0,1)
            outputs[1][idx] = current_mask * outputs[1][idx]
        probs = []
        for idx, hypothesis in enumerate(outputs[1]):
            ppl = 0.0
            for idx2, token in enumerate(hypothesis):
                if not len(token[token.nonzero()]):
                    continue
                ppl += token[input_ids[idx][idx2]].item()
                
            hypothesis_length = len([ x for x in hypothesis if len(x[x.nonzero()]) ])
            if hypothesis_length:
                ppl = np.float(torch.abs(torch.sum(item.view(-1))))
                ppl = ppl / hypothesis_length
            else:
                ppl = 0.0
            probs.append(1 / ppl)

        return probs
    
    def insert_padding(self,arr,padding):
        max_len = np.max([len(a) for a in arr])
        padded_arr = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=padding) for a in arr])
        return padded_arr

    def get_probs(self, list_of_lists):
        input_ids, attention_mask = self.batch_encode(list_of_lists)
        probs = self.calculate_loss(input_ids, attention_mask)
        return probs

class GPT2LM(object):

    def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=20):
        self.scorer = LMScorer.from_pretrained(model_name_or_path, device=device, batch_size=gpu_batch_size)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
        self.model.to(device)

    def greedy_generation(self, sentence, max_length=30, skip_special_tokens=True):
        input_ids = self.tokenizer.encode(sentence)
        greedy_output = self.model.generate(input_ids, max_length=max_length)
        result = self.tokenizer.decode(greedy_output[0], skip_special_tokens=skip_special_tokens)
        return result

    def get_best_candidate(self, list_of_candidates, search='greedy', expansions=10, **kwargs):
        scores = []
        for item in list_of_candidates:
            candidate_expansions = []
            for i in range(expansions):
                if search == 'greedy':
                    expansion = self.greedy_generation(item, **kwargs)
                else:
                    raise NotImplementedError
                candidate_expansions.append(expansion)
            candidate_scores = self.get_probs(candidate_expansions)
            candidate_average_scores = np.average(candidate_scores)
            scores.append(candidate_average_scores)
        chosen_candidate = list_of_candidates[np.argmin(scores)]
        return chosen_candidate

    def get_probs(self, list_of_candidates):
        scores =  self.scorer.sentence_score(list_of_candidates, log=True)
        scores = [ 1.0 - x for x in scores ]
        return scores

class ModelLM(object):

    def __init__(self, model_name_or_path=None, model_type=None, device=None, gpu_batch_size=None):
        if model_type == 'gpt2':
            self.model = GPT2LM(model_name_or_path, device=device, gpu_batch_size=gpu_batch_size)
        elif model_type == 'bert':
            self.model = BertLM(model_name_or_path, device=device, gpu_batch_size=gpu_batch_size)
        else:
            raise NotImplementedError

class Beamsearch(ModelLM):

    def __init__(self, dataset=None, model_name_or_path=None, model_type=None, device='cuda', gpu_batch_size=1):
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

    def run(self, topk=10, steps=2):
        self.tree = copy.deepcopy(self.dataset)
        for i in range(steps):

            self.tree = self.next_step(self.tree)

            self.reshape_tree(self.gpu_batch_size)
            self.update_probabilities()
            self.tree = self.flatten_list(self.tree)

            self.tree = self.trim_tree(topk)
        
        return self.convert_dataset()

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
    raise NotImplementedError
    return [], 0

def main():
    parser = HfArgumentParser((ModelArguments, DataEvaluationArguments, BeamsearchArguments))
    model_args, data_args, beam_args = parser.parse_args_into_dataclasses()
    reader = DatasetReader(data_args.eval_data_file, data_args.eval_dataset_format)
    reader.read()
    
    beamsearch = Beamsearch(
        dataset=reader.dataset, 
        model_name_or_path=model_args.model_name_or_path, 
        model_type=model_args.model_type,
        gpu_batch_size=model_args.gpu_batch_size)
    
    start_time = timer()
    segmented_data = beamsearch.run(beam_args.topk, beam_args.steps)
    end_time = timer()
    elapsed_time = end_time - start_time
    result = evaluate_word_segmentation_model(segmented_data, reader.test, save_report=True, report_file='report.json')

    print(result)
    print('characters / second: ', reader.character_count / elapsed_time)

    with open('dict.json', 'w+') as f:
        json.dump(beamsearch.prob_dict, f)
    
    if model_args.model_type == 'gpt2':
        groups, groups_character_count = gather_n_candidates(beamsearch.prob_dict, segmented_data, n=2)

        start_time = timer()
        new_segmented_data = [ beamsearch.model.get_best_candidate(group) for group in groups ]
        end_time = timer()
        elapsed_time = end_time - start_time
    
    new_result = evaluate_word_segmentation_model(new_segmented_data, reader.test, save_report=True, report_file='report.json')
    print(new_result)
    print('characters / second: ', groups_character_count / elapsed_time)

    

if __name__ == '__main__':
    main()