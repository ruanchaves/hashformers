import pandas as pd
import copy
import numpy as np 

from hashformers.experiments.evaluation import (
    filter_top_k
)

def project_scores(a, b, segmentation_field="segmentation", score_field="score"):
    """
    Projects the score from dataframe 'b' onto dataframe 'a' based on the segmentation_field.

    It first creates a view of dataframe 'b' with unique values of the segmentation_field.
    Then it merges this view with dataframe 'a' and updates the score field in 'a' with the score from 'b'.
    The resulting dataframe is sorted by score in ascending order.

    Args:
        a (pandas.DataFrame): The dataframe onto which the scores are to be projected.
        b (pandas.DataFrame): The dataframe from which the scores are taken.
        segmentation_field (str, optional): The field based on which scores are projected. Defaults to "segmentation".
        score_field (str, optional): The field which contains the scores to be projected. Defaults to "score".

    Returns:
        pandas.DataFrame: The dataframe 'a' with updated scores projected from dataframe 'b'.
    """
    b_view = b[[segmentation_field, score_field]]\
        .drop_duplicates(subset=[segmentation_field])
    df = pd.merge(a, b_view, on=segmentation_field, how='left')
    df = df.drop([score_field+'_x'], axis=1)
    df = df.rename(columns={
        score_field+'_y': score_field
    })
    df = df.sort_values(by=score_field, ascending=True)
    return df

def filter_and_project_scores(a, b, characters_field="hashtag", segmentation_field="segmentation"):
    """
    Filters the top two records of the dataframe 'a', projects the scores from dataframe 'b' onto 'a',
    and returns both the modified dataframes.

    Args:
        a (pandas.DataFrame): The first dataframe, which will be filtered and onto which the scores will be projected.
        b (pandas.DataFrame): The second dataframe, from which the scores are taken.
        characters_field (str, optional): The field used to sort the dataframes. Defaults to "hashtag".
        segmentation_field (str, optional): The field based on which scores are projected. Defaults to "segmentation".

    Returns:
        list of pandas.DataFrame: The modified dataframes 'a' and 'b' after filtering and projecting scores.
    """
    models = copy.deepcopy([a,b])
    for idx, m in enumerate(models):
        models[idx] = models[idx]\
            .sort_values(by=[characters_field, segmentation_field])

    models[0] = filter_top_k(models[0], 2, fill=True)
    models[1] = project_scores(models[0], models[1])

    for idx, m in enumerate(models):
        models[idx] = models[idx]\
            .sort_values(by=[characters_field, segmentation_field])\
            .reset_index(drop=True)
    return models

def calculate_diff_scores(a, b, characters_field="hashtag", score_field="score"):
    """
    Calculates the difference in scores between pairs of records in the dataframes 'a' and 'b'.

    Args:
        a (pandas.DataFrame): The first dataframe.
        b (pandas.DataFrame): The second dataframe.
        characters_field (str, optional): The field used to sort the dataframes. Defaults to "hashtag".
        score_field (str, optional): The field which contains the scores. Defaults to "score".

    Returns:
        list of pandas.DataFrame: The modified dataframes 'a' and 'b' with an additional 'diff' column indicating the score difference.
    """
    models = copy.deepcopy([a,b])
    for idx, m in enumerate(models):
        
        models[idx] = models[idx]\
            .sort_values(by=[characters_field, score_field])
        score_pairs = models[idx][score_field].values.reshape(-1,2)

        models[idx]['rank'] = \
            score_pairs.argsort().flatten()
        models[idx]['diff'] = \
            np.repeat(np.subtract.reduce(score_pairs, axis=1).flatten(), 2)
        models[idx]['diff'] = \
            models[idx]['diff'].fillna(0.0)
    return models

def build_ensemble_df(a, b):
    """
    Builds an ensemble dataframe from the input dataframes 'a' and 'b'.

    It filters and projects the scores from 'b' onto 'a', calculates the score differences,
    and then merges the differences back into the 'a' dataframe. 

    Args:
        a (pandas.DataFrame): The first dataframe.
        b (pandas.DataFrame): The second dataframe.

    Returns:
        pandas.DataFrame: The resulting ensemble dataframe with projected scores and score differences.
    """
    models = filter_and_project_scores(a, b)
    models = calculate_diff_scores(models[0], models[1])
    
    for idx, m in enumerate(models):
        models[idx]['diff'] = np.abs(models[idx]['diff'].values)

    models[0]['diff_2'] = models[1]['diff'] 
    models[0]['rank_2'] = models[1]['rank']

    return models[0]