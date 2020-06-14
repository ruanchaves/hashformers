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
from reader import DatasetReader
from configuration_classes import (
    parameters_to_string,
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

import logging
import logging.config

import os
import pathlib
import itertools 

def format_expansions(expansions_list):
    expansions_list = itertools.chain(*expansions_list)
    formatted_result_list = []
    for item in expansions_list:
        hypothesis = item['hypothesis'].split('=')[0].strip()
        generation = item['generation'].split("=")[1].strip()
        output = ''
        for character in generation:
            if len(output.replace(" ", "")) == len(hypothesis):
                break
            output += character
        if len(output) < len(hypothesis):
            output = hypothesis
        formatted_result_list.append(output)
    return formatted_result_list

def main():

    parser = HfArgumentParser((ModelArguments, DataEvaluationArguments, BeamsearchArguments, RankingArguments))
    model_args, data_args, beam_args, ranking_args = parser.parse_args_into_dataclasses()

    logging.config.fileConfig('logging.conf', \
        defaults={'logfilename': data_args.logfile})
    logger = logging.getLogger(__file__)

    if os.path.isfile(data_args.dict_file):
        sys.exit()

    logger.info('\n' + parameters_to_string(model_args, data_args, beam_args, ranking_args))
    reader = DatasetReader(data_args.eval_data_file, data_args.eval_dataset_format)
    reader.read()
    
    gpt2_lm = GPT2LM(model_args.model_name_or_path, device='cuda', gpu_batch_size=model_args.gpu_batch_size)
    character_count = sum([len(x) for x in reader.dataset])
    input_data = ["{0} = ".format(x) for x in reader.dataset]
    start_time = timer()
    gpt2_lm.greedy_generation(input_data)
    end_time = timer()

    elapsed_time = end_time - start_time
    logger.info('elapsed time: {0}'.format(elapsed_time))    
    logger.info('characters / second: {0}'.format(character_count / elapsed_time))
    result = format_expansions(gpt2_lm.expansions)
    result = { 'hypothesis': item, 'value': idx for idx, item in enumerate(result)}

    dict_dir = os.path.split(data_args.dict_file)[0]
    pathlib.Path(dict_dir).mkdir(parents=True, exist_ok=True)
    with open(data_args.dict_file, 'w+') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main()