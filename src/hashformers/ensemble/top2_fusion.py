from hashformers.beamsearch.data_structures import enforce_prob_dict
from hashformers.experiments.utils import build_ensemble_df

def run_ensemble(
    a_diff,
    b_diff,
    a_rank,
    b_rank,
    alpha=0.0,
    beta=0.0):
    """
    Computes the ensemble output using given differences and ranks with weights alpha and beta.

    Args:
        a_diff (array-like): Differences corresponding to 'a'.
        b_diff (array-like): Differences corresponding to 'b'.
        a_rank (array-like): Ranks corresponding to 'a'.
        b_rank (array-like): Ranks corresponding to 'b'.
        alpha (float, optional): The weight for 'a_diff'. Default is 0.0.
        beta (float, optional): The weight for 'b_diff'. Default is 0.0.

    Returns:
        array-like: An array-like object representing the ensemble output.
    """
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
    """
    Computes the ensemble of two given dictionaries using the specified weights alpha and beta.

    Args:
        dict_1 (dict): The first input dictionary.
        dict_2 (dict): The second input dictionary.
        alpha (float, optional): The weight for differences in 'dict_1'. Default is 0.2.
        beta (float, optional): The weight for differences in 'dict_2'. Default is 0.1.

    Returns:
        DataFrame: A pandas DataFrame representing the ensemble of the two input dictionaries.
    """
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

class Top2_Ensembler(object):
    """
    A class that provides a method to run the ensemble of a segmenter run and a reranker run.

    Args:
        None
    """

    def __init__(self):
        pass

    def run(self, segmenter_run, reranker_run, alpha=0.222, beta=0.111):
        """
        Runs the ensemble of a segmenter run and a reranker run.

        Args:
            segmenter_run (dict or ProbabilityDictionary): The result of a segmenter run.
            reranker_run (dict or ProbabilityDictionary): The result of a reranker run.
            alpha (float, optional): The weight for differences in 'segmenter_run'. Default is 0.222.
            beta (float, optional): The weight for differences in 'reranker_run'. Default is 0.111.

        Returns:
            ProbabilityDictionary: A ProbabilityDictionary representing the ensemble of the segmenter and reranker runs.
        """
        ensemble = top2_ensemble(
            segmenter_run,
            reranker_run,
            alpha=alpha,
            beta=beta
        )

        ensemble_prob_dict = enforce_prob_dict(
            ensemble,
            score_field="ensemble_rank")
        
        return ensemble_prob_dict