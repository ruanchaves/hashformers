import pandas as pd
import copy
import numpy as np 

from word_segmentation.evaluation.utils import (
    filter_top_k
)

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

def filter_and_project_scores(a,b):
    models = copy.deepcopy([a,b])
    for idx, m in enumerate(models):
        models[idx] = models[idx].sort_values(by=['hashtag', 'segmentation'])

    models[0] = filter_top_k(models[0], 2, fill=True)
    models[1] = project_scores(models[0], models[1])

    for idx, m in enumerate(models):
        models[idx] = models[idx].sort_values(by=['hashtag', 'segmentation']).reset_index(drop=True)
    return models

def calculate_diff_scores(a, b):
    models = copy.deepcopy([a,b])
    for idx, m in enumerate(models):
        
        models[idx] = models[idx].sort_values(by=['hashtag', 'score'])
        score_pairs =  models[idx]['score'].values.reshape(-1,2)

        models[idx]['rank'] = score_pairs.argsort().flatten()
        models[idx]['diff'] = np.repeat(np.subtract.reduce(score_pairs, axis=1).flatten(), 2)
        models[idx]['diff'] = models[idx]['diff'].fillna(0.0)
    return models

def build_ensemble_df(
    gpt2,
    bert
):
    models = filter_and_project_scores(gpt2, bert)
    models = calculate_diff_scores(*models)
    
    for idx, m in enumerate(models):
        models[idx]['diff'] = np.abs(models[idx]['diff'].values)

    models[0]['diff_2'] = models[1]['diff'] 
    models[0]['rank_2'] = models[1]['rank']

    return models[0]