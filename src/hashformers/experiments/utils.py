import pandas as pd
import copy
import numpy as np 

from hashformers.utils.filtering import filter_top_k

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

def filter_and_project_scores(a, b, k=2, characters_field="hashtag", segmentation_field="segmentation"):
    """
    Filters the top k records of the dataframe 'a', projects the scores from dataframe 'b' onto 'a',
    and returns both the modified dataframes.

    Args:
        a (pandas.DataFrame): The first dataframe, which will be filtered and onto which the scores will be projected.
        b (pandas.DataFrame): The second dataframe, from which the scores are taken.
        k (int, optional): The number of top records to retain for each group. Defaults to 2.
        characters_field (str, optional): The field used to sort the dataframes. Defaults to "hashtag".
        segmentation_field (str, optional): The field based on which scores are projected. Defaults to "segmentation".

    Returns:
        list of pandas.DataFrame: The modified dataframes 'a' and 'b' after filtering and projecting scores.
    """
    models = copy.deepcopy([a,b])
    for idx, m in enumerate(models):
        models[idx] = models[idx]\
            .sort_values(by=[characters_field, segmentation_field])

    models[0] = filter_top_k(models[0], k, fill=True)
    models[1] = project_scores(models[0], models[1])

    for idx, m in enumerate(models):
        models[idx] = models[idx]\
            .sort_values(by=[characters_field, segmentation_field])\
            .reset_index(drop=True)
    return models

def calculate_diff_scores(a, b, k=2, characters_field="hashtag", score_field="score"):
    """
    Calculates the difference in scores between pairs of records in the dataframes 'a' and 'b'.

    Note: This function is specific to k=2 (pairwise comparison). For k>2, use 
    calculate_weighted_scores instead.

    Args:
        a (pandas.DataFrame): The first dataframe.
        b (pandas.DataFrame): The second dataframe.
        k (int, optional): The number of candidates per group. Must be 2 for this function. Defaults to 2.
        characters_field (str, optional): The field used to sort the dataframes. Defaults to "hashtag".
        score_field (str, optional): The field which contains the scores. Defaults to "score".

    Returns:
        list of pandas.DataFrame: The modified dataframes 'a' and 'b' with an additional 'diff' column indicating the score difference.
    
    Raises:
        ValueError: If k != 2. For k>2, use calculate_weighted_scores instead.
    """
    if k != 2:
        raise ValueError(
            f"calculate_diff_scores only supports k=2 (pairwise comparison). "
            f"Got k={k}. For k>2, use calculate_weighted_scores instead."
        )
    
    models = copy.deepcopy([a,b])
    for idx, m in enumerate(models):
        
        models[idx] = models[idx]\
            .sort_values(by=[characters_field, score_field])
        score_pairs = models[idx][score_field].values.reshape(-1, k)

        models[idx]['rank'] = \
            score_pairs.argsort().flatten()
        models[idx]['diff'] = \
            np.repeat(np.subtract.reduce(score_pairs, axis=1).flatten(), k)
        models[idx]['diff'] = \
            models[idx]['diff'].fillna(0.0)
    return models


def calculate_weighted_scores(a, b, k, alpha=1.0, beta=1.0, characters_field="hashtag", score_field="score"):
    """
    Calculates weighted fusion scores for top-k candidates.

    For each candidate, computes: final_score = alpha * a.score + beta * b.score
    Then assigns ranks based on the fused scores within each group.

    This is the generalized scoring strategy for k>=2 candidates, replacing the
    pairwise difference approach used in calculate_diff_scores.

    Args:
        a (pandas.DataFrame): The first dataframe (e.g., segmenter output).
        b (pandas.DataFrame): The second dataframe (e.g., reranker output).
        k (int): The number of candidates per group.
        alpha (float, optional): Weight for scores from dataframe 'a'. Defaults to 1.0.
        beta (float, optional): Weight for scores from dataframe 'b'. Defaults to 1.0.
        characters_field (str, optional): The field used to define groups. Defaults to "hashtag".
        score_field (str, optional): The field containing scores. Defaults to "score".

    Returns:
        pandas.DataFrame: A dataframe with weighted fusion scores and ranks.
            Contains columns: original columns from 'a', 'score_a', 'score_b', 
            'weighted_score', and 'weighted_rank'.
    """
    models = copy.deepcopy([a, b])
    
    for idx, m in enumerate(models):
        models[idx] = models[idx]\
            .sort_values(by=[characters_field, score_field])\
            .reset_index(drop=True)
    
    # Merge scores from both models
    result = models[0].copy()
    result = result.rename(columns={score_field: 'score_a'})
    
    # Add scores from model b
    result['score_b'] = models[1][score_field].values
    
    # Calculate weighted fusion score (lower is better, consistent with existing logic)
    result['weighted_score'] = alpha * result['score_a'] + beta * result['score_b']
    
    # Assign ranks within each group based on weighted score
    result['weighted_rank'] = result.groupby(characters_field)['weighted_score']\
        .rank(method='first', ascending=True).astype(int) - 1
    
    return result

def build_ensemble_df(a, b, k=2):
    """
    Builds an ensemble dataframe from the input dataframes 'a' and 'b'.

    It filters and projects the scores from 'b' onto 'a', calculates the score differences,
    and then merges the differences back into the 'a' dataframe.

    For k=2, uses pairwise difference scoring (original behavior).
    For k>2, uses weighted fusion scoring.

    Args:
        a (pandas.DataFrame): The first dataframe.
        b (pandas.DataFrame): The second dataframe.
        k (int, optional): Number of top candidates per group. Defaults to 2.

    Returns:
        pandas.DataFrame: The resulting ensemble dataframe with projected scores and score differences.
    """
    models = filter_and_project_scores(a, b, k=k)
    models = calculate_diff_scores(models[0], models[1], k=k)
    
    for idx, m in enumerate(models):
        models[idx]['diff'] = np.abs(models[idx]['diff'].values)

    models[0]['diff_2'] = models[1]['diff'] 
    models[0]['rank_2'] = models[1]['rank']

    return models[0]


def build_ensemble_df_topk(a, b, k, alpha=0.5, beta=0.5):
    """
    Builds an ensemble dataframe from the input dataframes 'a' and 'b' using
    weighted sum fusion for k>=2 candidates.

    This is the generalized version that works for any k, using weighted score
    fusion instead of pairwise difference.

    Args:
        a (pandas.DataFrame): The first dataframe (e.g., segmenter output).
        b (pandas.DataFrame): The second dataframe (e.g., reranker output).
        k (int): Number of top candidates per group.
        alpha (float, optional): Weight for segmenter scores. Defaults to 0.5.
        beta (float, optional): Weight for reranker scores. Defaults to 0.5.

    Returns:
        pandas.DataFrame: The resulting ensemble dataframe with weighted scores and ranks.
    """
    models = filter_and_project_scores(a, b, k=k)
    result = calculate_weighted_scores(
        models[0], models[1], 
        k=k, 
        alpha=alpha, 
        beta=beta
    )
    return result