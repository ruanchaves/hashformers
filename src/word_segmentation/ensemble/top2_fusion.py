from word_segmentation.beamsearch.data_structures import enforce_prob_dict
from word_segmentation.experiments.architectures import build_ensemble_df, run_ensemble

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