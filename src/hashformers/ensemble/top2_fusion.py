from hashformers.beamsearch.data_structures import enforce_prob_dict
from hashformers.experiments.utils import build_ensemble_df

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

def top2_ensemble(
    dict_1, 
    dict_2, 
    alpha=0.2, 
    beta=0.1):

    a = enforce_prob_dict(dict_1).to_dataframe(
        characters_field="hashtag"
    )
    b = enforce_prob_dict(dict_2).to_dataframe(
        characters_field="hashtag"
    )

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

    return ensemble_df