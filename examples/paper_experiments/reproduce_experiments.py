from word_segmentation.experiments.architectures import grid_search_and_evaluate
import json
import copy
from argparse import ArgumentParser
import logging
import datetime
import os
import pandas as pd
import pathlib 

from word_segmentation.evaluation.utils import (
    evaluate_df,
    filter_top_k
)

def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--source',
        type=str,
        default='./data.json'
    )

    args = parser.parse_args()

    return args

def configure_logging():
    pathlib.Path('./logs').mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("word_segmentation_experiments")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    timestamp = int(datetime.datetime.now().timestamp())
    fh = logging.FileHandler(
        os.path.join("./logs", f"word_segmentation_experiments_{timestamp}.log")
        )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def log_dataframe(dataframe, logger_name="word_segmentation_experiments"):
    logger = logging.getLogger(logger_name)
    logger.info('\n\t'+dataframe.to_string().replace('\n', '\n\t'))

def evaluate_single_runs(data, k=1):
    evaluate_dataset = lambda x: evaluate_df(filter_top_k(x, k))
    eval_df = copy.deepcopy(data)
    eval_df = [ x for x in eval_df if x['model'] ]
    for idx, item in enumerate(eval_df):
        current_run = pd.DataFrame(eval_df[idx]['data'])
        metrics = evaluate_dataset(current_run)
        eval_df[idx].update(metrics)
        del eval_df[idx]['data']
    eval_df = pd.DataFrame(eval_df)
    eval_df = eval_df\
        .sort_values(
            by=[
                'dataset', 
                'model', 
                'f1', 
                'acc'], 
                ascending=False)\
        .set_index(
            [
                'dataset', 
                'model'])\
        .applymap(lambda x: round(x, 3))
    return eval_df

def main():
    args = get_args()
    logger = configure_logging()
    with open(args.source, 'r') as f:
        data = json.load(f)
    
    for k in range(1, 10):
        logger.info(f'k = {k}')
        evaluation = evaluate_single_runs(data, k=k)
        log_dataframe(evaluation)

    boun_test_metrics = grid_search_and_evaluate(
        data,
        dev_set="Dev-BOUN",
        test_set="Test-BOUN",
        bert_model="bert_from_gpt2",
        gpt2_model="gpt2"
    )

    logger.info("%s", boun_test_metrics)

    stanford_test_metrics = grid_search_and_evaluate(
        data,
        dev_set="Dev-Stanford",
        test_set="Test-Stanford",
        bert_model="bert_from_gpt2",
        gpt2_model="gpt2"
    )

    logger.info("%s", stanford_test_metrics)


if __name__ == '__main__':
    main()