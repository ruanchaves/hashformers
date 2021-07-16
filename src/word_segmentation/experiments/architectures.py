from word_segmentation.experiments.utils import build_ensemble_df
from word_segmentation.experiments.evaluation import (
    read_experiment_dataset,
    evaluate_df,
    filter_top_k
)
import numpy as np 
import itertools
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import copy

def run_ensemble(
    a_diff,
    b_diff,
    a_rank,
    b_rank,
    alpha=0.0,
    beta=0.0):

    delta = alpha * a_diff - beta * b_diff
    decision = (delta < 0).astype(int)
    negation =  (~(delta < 0)).astype(int)
    output = a_rank * negation + b_rank * decision
    
    return output

def evaluate_ensemble(
    input_df,
    alpha=0.0,
    beta=0.0):
    
    df = copy.deepcopy(input_df)

    df['ensemble_rank'] = run_ensemble(
        df['diff'].values,
        df['diff_2'].values,
        df['rank'].values,
        df['rank_2'].values,
        alpha=alpha,
        beta=beta
    )

    df_1 = filter_top_k(
        df,
        1,
        characters_field="gold",
        score_field="ensemble_rank"
    )

    metrics_df_1 = evaluate_df(df_1)
    metrics_df_1.update({'k': 1})
    metrics_df_2 = evaluate_df(df)
    metrics_df_2.update({'k': 2})

    metrics_df_2 = {
        f'top2_{key}':value for key,value in metrics_df_2.items()
    }

    metrics = {}
    metrics.update(metrics_df_1)
    metrics.update(metrics_df_2)
    return metrics

def build_ensemble_df_from_data(
    data,
    dataset,
    aux_model='bert_from_gpt2',
    ref_model='gpt2'
):
    aux = read_experiment_dataset(
        data,
        dataset,
        aux_model
    )

    ref = read_experiment_dataset(
        data,
        dataset,
        ref_model
    )

    assert aux.shape == ref.shape

    ensemble_df = build_ensemble_df(ref, aux)

    return ensemble_df

def grid_search(
    data,
    dataset,
    bert_model='bert_from_gpt2',
    gpt2_model='gpt2',
    num_params=10
):
    ensemble_df = build_ensemble_df_from_data(
        data,
        dataset,
        ref_model=bert_model,
        aux_model=gpt2_model
    )

    alpha = np.linspace(0.0, 1.0, num=num_params).round(3)
    beta = np.linspace(0.0, 1.0, num=num_params).round(3)

    params = list(itertools.product(alpha, beta))

    def process(item):
        metrics = evaluate_ensemble(
            ensemble_df,
            alpha=item[0],
            beta=item[1]
        )
        output = {
            'alpha': item[0],
            'beta': item[1]
        }
        output.update(metrics)
        return output

    results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process)(item) for item in params)
    results = pd.DataFrame(results)
    results = results.sort_values(by=['f1', 'acc'], ascending=False)
    return results

def grid_search_and_evaluate(
    data,
    dev_set='Dev-BOUN',
    test_set='Test-BOUN',
    bert_model='bert_from_gpt2',
    gpt2_model='gpt2'
):

    dev_report = grid_search(
        data,
        dev_set,
        bert_model=bert_model,
        gpt2_model=gpt2_model
    )

    alpha = dev_report.iloc[0, :]['alpha'] 
    beta = dev_report.iloc[0, :]['beta']

    ensemble_df = build_ensemble_df_from_data(
        data,
        test_set,
        ref_model=bert_model,
        aux_model=gpt2_model
    )

    ensemble_metrics = evaluate_ensemble(
        ensemble_df,
        alpha=alpha,
        beta=beta
    )

    metrics = {
        "bert_model": bert_model,
        "gpt2_model": gpt2_model,
        "dev_set": dev_set,
        "test_set": test_set,
        "alpha": alpha,
        "beta": beta
    }

    metrics.update(ensemble_metrics)

    return metrics
