import pandas as pd
import copy
import numpy as np 

from word_segmentation.experiments.evaluation import (
    filter_top_k
)

def project_scores(
    a, 
    b, 
    segmentation_field="segmentation", 
    score_field="score"):
  b_view = b[[segmentation_field, score_field]]\
      .drop_duplicates(subset=[segmentation_field])
  df = pd.merge(a, b_view, on=segmentation_field, how="left")
  df = df.drop([score_field+"_x"], axis=1)
  df = df.rename(columns={
      score_field+"_y": score_field
  })
  df = df.sort_values(by=score_field, ascending=True)
  return df

def filter_and_project_scores(
    a,
    b,
    characters_field="hashtag",
    segmentation_field="segmentation",
    score_field="score",
    group_length_field="group_length",
    fill=True):
    models = copy.deepcopy([a,b])
    for idx, m in enumerate(models):
        models[idx] = models[idx].sort_values(by=[
            characters_field,
            segmentation_field
        ])

    models[0] = filter_top_k(
        models[0], 
        2, 
        characters_field=characters_field,
        score_field=score_field,
        segmentation_field=segmentation_field,
        group_length_field=group_length_field,
        fill=fill)

    models[1] = project_scores(
        models[0], 
        models[1],
        segmentation_field=segmentation_field,
        score_field=score_field)

    for idx, m in enumerate(models):
        models[idx] = models[idx].sort_values(by=[
            characters_field, 
            segmentation_field]).reset_index(drop=True)
    return models

def calculate_diff_scores(
    a, 
    b,
    characters_field="hashtag",
    score_field="score",
    rank_field="rank",
    diff_field="diff"):
    models = copy.deepcopy([a,b])
    for idx, m in enumerate(models):
        
        models[idx] = models[idx].sort_values(by=[
            characters_field,
            score_field])
        score_pairs =  models[idx][score_field].values.reshape(-1,2)

        models[idx][rank_field] = \
            score_pairs.argsort().flatten()
        models[idx][diff_field] = \
            np.repeat(np.subtract.reduce(score_pairs, axis=1).flatten(), 2)
        models[idx][diff_field] = \
            models[idx][diff_field].fillna(0.0)
    return models

def build_ensemble_df(
    ref_model_df,
    aux_model_df,
    ref_diff_field="diff",
    aux_diff_field="diff_2",
    ref_rank_field="rank",
    aux_rank_field="rank_2",
    characters_field="hashtag",
    segmentation_field="segmentation",
    score_field="score",
    group_length_field="group_length",
    fill=True
):
    models = filter_and_project_scores(
        ref_model_df, 
        aux_model_df,
        characters_field=characters_field,
        segmentation_field=segmentation_field,
        score_field=score_field,
        group_length_field=group_length_field,
        fill=fill)
    models = calculate_diff_scores(
        models[0],
        models[1],
        characters_field=characters_field,
        score_field=score_field,
        rank_field=ref_rank_field,
        diff_field=ref_diff_field
    )

    for idx, m in enumerate(models):
        models[idx][ref_diff_field] = \
            np.abs(models[idx][ref_diff_field].values)

    models[0][aux_diff_field] = models[1][ref_diff_field] 
    models[0][aux_rank_field] = models[1][ref_rank_field]

    return models[0]