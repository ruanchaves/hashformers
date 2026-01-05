"""Ensemble module for hashformers."""

from hashformers.ensemble.top2_fusion import (
    TopKEnsembler,
    Top2Ensembler,
    Top2_Ensembler,  # Backwards compatibility alias
    topk_ensemble,
    top2_ensemble,
    run_ensemble,
    run_weighted_ensemble,
)

__all__ = [
    "TopKEnsembler",
    "Top2Ensembler",
    "Top2_Ensembler",  # Deprecated alias for backwards compatibility
    "topk_ensemble",
    "top2_ensemble",
    "run_ensemble",
    "run_weighted_ensemble",
]
