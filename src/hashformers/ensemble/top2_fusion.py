from hashformers.beamsearch.data_structures import enforce_prob_dict
from hashformers.experiments.utils import build_ensemble_df, build_ensemble_df_topk


def run_weighted_ensemble(scores, ranks, k):
    """
    Selects the best candidate (rank 0) for each group from top-k weighted scores.

    For each group of k candidates, returns the segmentation with the lowest weighted rank.

    Args:
        scores (array-like): Weighted scores for all candidates.
        ranks (array-like): Weighted ranks (0 to k-1) for all candidates.
        k (int): Number of candidates per group.

    Returns:
        array-like: Array of selected segmentation indices (best rank per group).
    """
    import numpy as np
    # Reshape to groups of k, find the index of rank 0 (best) in each group
    ranks_reshaped = ranks.reshape(-1, k)
    # Get the local index (0 to k-1) of the best candidate in each group
    best_indices = ranks_reshaped.argmin(axis=1)
    return best_indices


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

class TopKEnsembler:
    """
    A class that provides a method to run the ensemble of a segmenter run and a reranker run
    using top-k weighted fusion.

    This generalized ensembler supports any k>=2 candidates, using weighted sum fusion
    to combine scores from multiple models.

    For k=2, this produces results equivalent to Top2Ensembler (backward compatible).
    For k>2, uses weighted fusion: final_score = alpha * segmenter_score + beta * reranker_score.

    Introduced in HASH-410 to generalize the Top-2 fusion logic.

    Args:
        k (int, optional): Number of top candidates to consider. Defaults to 2.
    
    Example:
        >>> ensembler = TopKEnsembler(k=5)
        >>> result = ensembler.run(segmenter_output, reranker_output, alpha=0.5, beta=0.5)
    """

    def __init__(self, k=2):
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}")
        self.k = k

    def run(self, segmenter_run, reranker_run, alpha=0.5, beta=0.5):
        """
        Runs the ensemble of a segmenter run and a reranker run.

        For k=2, uses the original pairwise difference logic for backward compatibility.
        For k>2, uses weighted sum fusion.

        Args:
            segmenter_run (dict or ProbabilityDictionary): The result of a segmenter run.
            reranker_run (dict or ProbabilityDictionary): The result of a reranker run.
            alpha (float, optional): Weight for segmenter scores. Default is 0.5.
            beta (float, optional): Weight for reranker scores. Default is 0.5.

        Returns:
            ProbabilityDictionary: A ProbabilityDictionary representing the ensemble of 
                the segmenter and reranker runs.
        """
        if self.k == 2:
            # Use original pairwise logic for k=2 (backward compatible)
            return self._run_top2(segmenter_run, reranker_run, alpha, beta)
        else:
            # Use weighted fusion for k>2
            return self._run_topk(segmenter_run, reranker_run, alpha, beta)

    def _run_top2(self, segmenter_run, reranker_run, alpha, beta):
        """Internal method for k=2 using pairwise difference."""
        ensemble = top2_ensemble(
            segmenter_run,
            reranker_run,
            alpha=alpha,
            beta=beta
        )
        ensemble_prob_dict = enforce_prob_dict(
            ensemble,
            score_field="ensemble_rank"
        )
        return ensemble_prob_dict

    def _run_topk(self, segmenter_run, reranker_run, alpha, beta):
        """Internal method for k>2 using weighted fusion."""
        a = enforce_prob_dict(segmenter_run).to_dataframe(
            characters_field="hashtag"
        )
        b = enforce_prob_dict(reranker_run).to_dataframe(
            characters_field="hashtag"
        )

        ensemble_df = build_ensemble_df_topk(a, b, k=self.k, alpha=alpha, beta=beta)
        
        ensemble_prob_dict = enforce_prob_dict(
            ensemble_df,
            score_field="weighted_rank"
        )
        return ensemble_prob_dict


def topk_ensemble(
    dict_1,
    dict_2,
    k=2,
    alpha=0.5,
    beta=0.5):
    """
    Computes the ensemble of two given dictionaries using weighted fusion for top-k candidates.

    This is the functional interface for TopKEnsembler.

    Args:
        dict_1 (dict): The first input dictionary (e.g., segmenter output).
        dict_2 (dict): The second input dictionary (e.g., reranker output).
        k (int, optional): Number of top candidates. Defaults to 2.
        alpha (float, optional): Weight for dict_1 scores. Defaults to 0.5.
        beta (float, optional): Weight for dict_2 scores. Defaults to 0.5.

    Returns:
        DataFrame: A pandas DataFrame representing the ensemble of the two input dictionaries.
    """
    a = enforce_prob_dict(dict_1).to_dataframe(
        characters_field="hashtag"
    )
    b = enforce_prob_dict(dict_2).to_dataframe(
        characters_field="hashtag"
    )

    if k == 2:
        # Use original pairwise logic for k=2
        return build_ensemble_df(a, b, k=k)
    else:
        # Use weighted fusion for k>2
        return build_ensemble_df_topk(a, b, k=k, alpha=alpha, beta=beta)


class Top2Ensembler(TopKEnsembler):
    """
    A class that provides a method to run the ensemble of a segmenter run and a reranker run.
    
    This is a specialized version of TopKEnsembler with k=2, using pairwise difference
    scoring for backward compatibility.

    Note: Renamed from Top2_Ensembler to Top2Ensembler for PEP8 compliance (HASH-012).
    
    For k>2 candidates, use TopKEnsembler directly.

    Args:
        None
    """

    def __init__(self):
        super().__init__(k=2)

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
        # Use original top2_ensemble for exact backward compatibility
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


# Backwards compatibility alias (HASH-012)
Top2_Ensembler = Top2Ensembler