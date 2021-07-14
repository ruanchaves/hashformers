from word_segmentation.beamsearch.data_structures import enforce_prob_dict
import numpy as np

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

def top2_ensemble(
    dict_1, 
    dict_2, 
    alpha=0.2, 
    beta=0.1,
    return_dataframe=False):

    a = enforce_prob_dict(dict_1)
    b = enforce_prob_dict(dict_2)

    reference_df = a.get_top_k(
        k=2,
        return_dataframe=True,
        fill=True
    )

    reference_df['ref_score'] = reference_df['score']
    reference_df['aux_score'] = reference_df['segmentation'].apply(
        lambda x: b.dictionary[x]
    )

    reference_df = reference_df.sort_values(
        by=['characters', 'ref_score']
    )

    reference_df['ref_diff'] = calculate_top2_diff(
        reference_df['ref_score'].values
    )

    reference_df = reference_df.sort_values(
        by=['characters', 'aux_score']
    )

    reference_df['aux_diff'] = calculate_top2_diff(
        reference_df['aux_score'].values
    )

    reference_df = reference_df.sort_values(
        by=["characters", "ref_score"]
    )

    reference_df["ref_rank"] = calculate_top2_rank(
        reference_df["ref_score"].values
    )
    reference_df["aux_rank"] = calculate_top2_rank(
        reference_df["aux_score"].values
    )

    ref_diff = reference_df["ref_diff"].values
    aux_diff = reference_df["aux_diff"].values
    ref_rank = reference_df["ref_rank"].values
    aux_rank = reference_df["aux_rank"].values

    reference_df['ensemble_rank'] = run_ensemble(
        ref_diff,
        aux_diff,
        ref_rank,
        aux_rank,
        alpha=alpha,
        beta=beta
    )

    reference_df = reference_df.sort_values(
        by=['segmentation', 'ensemble_rank']
    )

    if return_dataframe == True:
        return reference_df
    elif return_dataframe == False:
        reference_df = reference_df\
            .sort_values(by=["characters", "ensemble_rank", "score"])\
            .groupby("characters")\
            .head(1)
        segs = reference_df['segmentation'].values.tolist()
        chars = [ x.replace(" ", "") for x in segs ]
        output = {
            k:v for k,v in list(zip(chars, segs))
        }
        return output