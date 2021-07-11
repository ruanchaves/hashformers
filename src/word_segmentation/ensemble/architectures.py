from word_segmentation.ensemble.utils import build_ensemble_df
from word_segmentation.evaluation.utils import read_experiment_dataset
from word_segmentation.evaluation.utils import evaluate_df, filter_top_k
import numpy as np 
import itertools
import logging
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
    beta=0.0,
    threshold=0.0,
    mode='weighted'):
    
    if mode in ['weighted', 'weighted_threshold']:
        weighted_a_diff = alpha * a_diff
        weighted_b_diff = beta * b_diff
        delta = weighted_a_diff - weighted_b_diff

    if mode == 'weighted':
        decision = (delta < 0).astype(int)
        negation =  (~(delta < 0)).astype(int)
    elif mode == 'threshold':
        decision = (a_diff < threshold).astype(int)
        negation = (~(a_diff < threshold)).astype(int)
    elif mode == 'weighted_threshold':
        decision = ((a_diff < threshold) & (delta < 0)).astype(int)
        negation = (~((a_diff < threshold) & (delta < 0))).astype(int)
    else:
        raise NotImplementedError
    
    output = a_rank * negation + b_rank * decision
    
    return output

def evaluate_ensemble(
    input_df,
    alpha=0.0,
    beta=0.0,
    threshold=0.0,
    mode='weighted'):
    
    df = copy.deepcopy(input_df)

    df['ensemble_rank'] = run_ensemble(
        df['diff'].values,
        df['diff_2'].values,
        df['rank'].values,
        df['rank_2'].values,
        alpha=alpha,
        beta=beta,
        threshold=threshold,
        mode=mode
    )

    df = filter_top_k(
        df,
        1,
        gold_field='gold',
        score_field='ensemble_rank'
    )

    metrics = evaluate_df(df)
    return metrics

def build_ensemble_df_from_data(
    data,
    dataset,
    bert_model='bert_from_gpt2',
    gpt2_model='gpt2'
):
    bert = read_experiment_dataset(
        data,
        dataset,
        bert_model
    )

    gpt2 = read_experiment_dataset(
        data,
        dataset,
        gpt2_model
    )

    assert bert.shape == gpt2.shape

    ensemble_df = build_ensemble_df(gpt2, bert)

    return ensemble_df

def grid_search(
    data,
    dataset,
    bert_model='bert_from_gpt2',
    gpt2_model='gpt2'
):

    logger = logging.getLogger("word_segmentation_experiments")    

    ensemble_df = build_ensemble_df_from_data(
        data,
        dataset,
        bert_model=bert_model,
        gpt2_model=gpt2_model
    )

    alpha = np.linspace(0.0, 1.0, num=10).round(3)
    beta = np.linspace(0.0, 1.0, num=10).round(3)
    threshold = np.linspace(0.0, 500.0, num=10).round(3)
    
    # threshold_params = [ ([None, None, x], 'threshold') for x in threshold ]
    
    weighted_params = list(itertools.product(alpha, beta))
    weighted_params = [ ([x, y, None], 'weighted') for x,y in weighted_params]

    # weighted_threshold_params = list(itertools.product(alpha, beta, threshold))
    # weighted_threshold_params = [ ([x,y,z], 'weighted_threshold') for x,y,z in weighted_threshold_params]

    # params = threshold_params + weighted_params + weighted_threshold_params
    params = weighted_params

    def process(item):
        metrics = evaluate_ensemble(
            ensemble_df,
            alpha=item[0][0],
            beta=item[0][1],
            threshold=item[0][2],
            mode=item[1]
        )
        output = {
            'alpha': item[0][0],
            'beta': item[0][1],
            'threshold': item[0][2],
            'mode': item[1]
        }
        output.update(metrics)
        logger.debug(output)
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
    threshold = dev_report.iloc[0, :]['threshold']
    mode = dev_report.iloc[0, :]['mode']

    ensemble_df = build_ensemble_df_from_data(
        data,
        test_set,
        bert_model=bert_model,
        gpt2_model=gpt2_model
    )

    ensemble_metrics = evaluate_ensemble(
        ensemble_df,
        alpha=alpha,
        beta=beta,
        threshold=threshold,
        mode=mode
    )

    metrics = {
        "bert_model": bert_model,
        "gpt2_model": gpt2_model,
        "dev_set": dev_set,
        "test_set": test_set,
        "alpha": alpha,
        "beta": beta,
        "threshold": threshold,
        "mode": mode
    }

    metrics.update(ensemble_metrics)

    return metrics