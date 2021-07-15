from word_segmentation.beamsearch.data_structures import enforce_prob_dict
import numpy as np
import copy
import pandas as pd

def calculate_top2_rank(scores):
    x = scores.reshape(-1,2)
    x = x.argsort().flatten()
    return x

def calculate_top2_diff(scores):
    x = scores.reshape(-1,2)
    x = np.repeat(np.subtract.reduce(x, axis=1).flatten(), 2)
    x = np.nan_to_num(x)
    return x

def run_ensemble(
    a_diff,
    b_diff,
    a_rank,
    b_rank,
    alpha=0.2,
    beta=0.1):

    delta = alpha * a_diff - beta * b_diff
    decision = (delta < 0).astype(int)
    negation =  (~(delta < 0)).astype(int)
    output = a_rank * negation + b_rank * decision
    
    return output

def filter_top_k(
    input_df, 
    k, 
    gold_field='gold', 
    score_field='score',
    segmentation_field='segmentation',
    fill=False):
  
  df = copy.deepcopy(input_df)
  
  df = df\
    .sort_values(by=score_field, ascending=True)\
    .groupby(gold_field)\
    .head(k)

  if fill:
    df['group_length'] = df.groupby(gold_field)[segmentation_field].transform(len)
    df['group_length'] = df['group_length'] * -1 + k + 1
    len_array = df['group_length'].values
    
    df = df.drop(columns=['group_length'])
    records = np.array(df.to_dict('records'))
    cloned_records = list(np.repeat(records, len_array))
    df = pd.DataFrame(cloned_records)
    
    df = df\
      .sort_values(by=score_field, ascending=True)\
      .groupby(gold_field)\
      .head(k)
  return df

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

def calculate_diff_scores(
    a,
    character_field='characters',
    score_field='score',
    diff_field='diff'):
    a = a.sort_values(character_field, score_field)
    a[diff_field] = calculate_top2_diff(a[score_field].values)
    return a

def calculate_rank(
    a,
    character_field='characters',
    score_field='score',
    rank_field='rank'):    
    a = a.sort_values(character_field, score_field)
    a[rank_field] = calculate_top2_rank(a[score_field].values)
    return a


def build_ensemble_df(
    a,
    b
):
    a, b = filter_and_project_scores(a, b)

    a = calculate_diff_scores(a)
    b = calculate_diff_scores(b)
    
    a = calculate_rank(a)
    b = calculate_rank(b)

    a["diff"] = np.abs(a["diff"].values)
    b["diff"] = np.abs(b["diff"].values)

    a["diff_2"] = b["diff"] 
    a["rank_2"] = b["rank"]

    return a

def filter_and_project_scores(
    a,
    b,
    character_field='characters',
    segmentation_field='segmentation'):


    a = a.sort_values(by=[character_field, segmentation_field])
    b = b.sort_values(by=[character_field, segmentation_field])

    a = filter_top_k(a, 2, fill=True)
    b = project_scores(a, b)

    a = a\
        .sort_values(
            by=[character_field, segmentation_field])\
        .reset_index(drop=True)
    b = b\
        .sort_values(
            by=[character_field, segmentation_field])\
        .reset_index(drop=True)
    return a,b

def top2_ensemble(
    dict_1, 
    dict_2, 
    alpha=0.2, 
    beta=0.1,
    return_dataframe=False):

    a = enforce_prob_dict(dict_1).to_dataframe()
    b = enforce_prob_dict(dict_2).to_dataframe()

    ensemble_df = build_ensemble_df(a, b)

    ref_diff = ensemble_df["diff"].values
    aux_diff = ensemble_df["diff_2"].values
    ref_rank = ensemble_df["rank"].values
    aux_rank = ensemble_df["rank_2"].values

    ensemble_df["ensemble_rank"] = run_ensemble(
        ref_diff,
        aux_diff,
        ref_rank,
        aux_rank,
        alpha=alpha,
        beta=beta
    )

    if return_dataframe == True:
        return ensemble_df
    elif return_dataframe == False:
        reference_df = ensemble_df\
            .sort_values(by=["characters", "ensemble_rank", "score"])\
            .groupby("characters")\
            .head(1)
        segs = reference_df["segmentation"].values.tolist()
        chars = [ x.replace(" ", "") for x in segs ]
        output = {
            k:v for k,v in list(zip(chars, segs))
        }
        return output