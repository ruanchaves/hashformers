import pandas as pd
import copy
from pandas.testing import assert_series_equal
import numpy as np 

from word_segmentation.evaluation.utils import (
    filter_top_k
)

from dataclasses import dataclass

@dataclass
class ModelRank:
    dataframe: pd.DataFrame
    diff_matrix: np.ndarray
    rank_matrix: np.ndarray

def generate_diff_matrix(x):
    diff_matrix = np.subtract.outer(x,x)
    diff_matrix *= np.tri(*diff_matrix.shape) # fill upper triangle with zeros
    return diff_matrix

def project_scores(
    a, 
    b, 
    segmentation_field='segmentation', 
    score_field='score'):
  b_view = b[[segmentation_field, score_field]]\
      .drop_duplicates(subset=[segmentation_field])
  df = pd.merge(a, b_view, on=segmentation_field, how='left')
  df = df.drop([score_field+'_x'], axis=1)
  df = df.rename(columns={
      score_field+'_y': score_field
  })
  df = df.sort_values(by=score_field, ascending=True)
  return df

def build_ensemble_df(
    gpt2,
    bert,
    k=2
):
    models = copy.deepcopy([gpt2, bert])

    for idx, m in enumerate(models):
        models[idx] = models[idx].sort_values(by=['hashtag', 'segmentation'])

    assert_series_equal(models[0]['hashtag'], models[1]['hashtag'], check_index=False)
    assert_series_equal(models[0]['segmentation'], models[1]['segmentation'], check_index=False)

    models[0] = filter_top_k(models[0], k, fill=True)
    models[1] = project_scores(models[0], models[1])

    for idx, m in enumerate(models):
        models[idx] = models[idx].sort_values(by=['hashtag', 'segmentation']).reset_index(drop=True)

    assert_series_equal(models[0]['hashtag'], models[1]['hashtag'], check_index=False)
    assert_series_equal(models[0]['segmentation'], models[1]['segmentation'], check_index=False)

    if k == 2:
        for idx, m in enumerate(models):
            
            models[idx] = models[idx].sort_values(by=['hashtag', 'score'])
            score_pairs =  models[idx]['score'].values.reshape(-1,2)

            models[idx]['rank'] = score_pairs.argsort().flatten()
            models[idx]['diff'] = np.repeat(np.subtract.reduce(score_pairs, axis=1).flatten(), 2)
            models[idx]['diff'] = models[idx]['diff'].fillna(0.0)

            assert_negatives = models[idx][models[idx]['diff'] > 0]
            assert (models[idx]['diff'].values <= 0).all(), print(assert_negatives)

            models[idx]['diff'] = np.abs(models[idx]['diff'].values)

        models[0]['diff_2'] = models[1]['diff'] 
        models[0]['rank_2'] = models[1]['rank']

        return models[0]
    elif k > 2:
        output = []
        for idx, m in enumerate(models):
            models[idx] = models[idx].sort_values(by=['hashtag', 'score'])
            models[idx]['rank'] = models[idx]['score'].values.argsort()
            diff_matrix = generate_diff_matrix(models[idx]['score'].values)
            rank_matrix = diff_matrix.astype(bool).astype(int)
            model_rank = ModelRank(models[idx], diff_matrix, rank_matrix)
            output.append(model_rank)
        return output
    else:
        raise NotImplementedError