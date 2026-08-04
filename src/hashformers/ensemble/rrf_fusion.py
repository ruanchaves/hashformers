"""Full-list reciprocal-rank fusion for segmentation candidates."""

import math
from collections.abc import Mapping, Sequence

from hashformers.beamsearch.data_structures import (
    ProbabilityDictionary,
    enforce_prob_dict,
)


def _validate_options(rrf_k, weights):
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, (int, float)):
        raise ValueError("rrf_k must be a finite nonnegative number")
    rrf_k = float(rrf_k)
    if not math.isfinite(rrf_k) or rrf_k < 0:
        raise ValueError("rrf_k must be a finite nonnegative number")
    if len(weights) == 0:
        raise ValueError("fusion weights must not be empty")
    validated = []
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("fusion weights must be finite nonnegative numbers")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("fusion weights must be finite nonnegative numbers")
        validated.append(weight)
    if not any(validated):
        raise ValueError("fusion weights must not all be zero")
    return rrf_k, validated


def _competition_ranks(probability_dictionary):
    """Return one-based competition ranks grouped by normalized input."""
    frame = (
        enforce_prob_dict(probability_dictionary)
        .to_dataframe()
        .reset_index(drop=True)
    )
    result = {}
    for characters, group in frame.groupby("characters", sort=False):
        group = group.sort_values("score", kind="stable")
        ranks = group["score"].rank(method="min", ascending=True).astype(int)
        result[characters] = list(zip(group["segmentation"], ranks))
    return result


def reciprocal_rank_fusion(rankings: Sequence, weights=None, rrf_k=60):
    """Fuse complete rankings with ``sum(weight / (k + rank))``.

    Component ranks are one-based competition ranks. Missing candidates add
    zero, fused ties fall back to component ranks and stable input order, and
    scores are negated to preserve Hashformers' lower-is-better convention.
    """
    if (
        not isinstance(rankings, Sequence)
        or isinstance(rankings, (str, bytes))
        or not rankings
    ):
        raise ValueError("rankings must be a non-empty sequence")
    if weights is None:
        weights = [1.0] * len(rankings)
    if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)):
        raise ValueError("fusion weights must be a sequence")
    if len(weights) != len(rankings):
        raise ValueError("fusion weights must match the number of rankings")
    rrf_k, weights = _validate_options(rrf_k, weights)
    component_ranks = [_competition_ranks(ranking) for ranking in rankings]
    fused = {}
    character_order = []
    for component in component_ranks:
        for characters in component:
            if characters not in character_order:
                character_order.append(characters)
    for characters in character_order:
        candidate_order = []
        rank_maps = []
        for component in component_ranks:
            entries = component.get(characters, [])
            rank_map = dict(entries)
            rank_maps.append(rank_map)
            for candidate, _ in entries:
                if candidate not in candidate_order:
                    candidate_order.append(candidate)
        scored = []
        for stable_position, candidate in enumerate(candidate_order):
            score = sum(
                weight / (rrf_k + rank_map[candidate])
                for weight, rank_map in zip(weights, rank_maps)
                if candidate in rank_map
            )
            fallbacks = tuple(
                rank_map.get(candidate, math.inf) for rank_map in rank_maps
            )
            scored.append((candidate, -score, fallbacks, stable_position))
        scored.sort(key=lambda item: (item[1], *item[2], item[3]))
        for candidate, score, _, _ in scored:
            fused[candidate] = score
    return ProbabilityDictionary(fused)


class ReciprocalRankFusionEnsembler:
    """Adapt segmenter and reranker outputs to full-list weighted RRF."""

    def run(self, segmenter_run, reranker_run, rrf_k=60, fusion_weights=None):
        if fusion_weights is None:
            fusion_weights = {"segmenter": 1.0, "reranker": 1.0}
        if not isinstance(fusion_weights, Mapping):
            raise ValueError("fusion_weights must be an object")
        unknown = set(fusion_weights) - {"segmenter", "reranker"}
        if unknown or set(fusion_weights) != {"segmenter", "reranker"}:
            raise ValueError("fusion_weights must contain segmenter and reranker")
        return reciprocal_rank_fusion(
            [segmenter_run, reranker_run],
            weights=[fusion_weights["segmenter"], fusion_weights["reranker"]],
            rrf_k=rrf_k,
        )
