"""Public Hashformers API with dependency-lazy exports.

Importing the package must stay lightweight so entry points such as the MCP
server can complete their protocol handshake before the Transformer runtime is
needed.  Public objects retain their historical top-level import locations and
are loaded on first attribute access.
"""

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "Any": ("typing", "Any"),
    "BaseSegmenter": (
        "hashformers.segmenter.base_segmenter",
        "BaseSegmenter",
    ),
    "BaseWordSegmenter": (
        "hashformers.segmenter.segmenter",
        "BaseWordSegmenter",
    ),
    "Beamsearch": ("hashformers.beamsearch.algorithm", "Beamsearch"),
    "DEFAULT_MAX_BATCH_SIZE": (
        "hashformers.batching",
        "DEFAULT_MAX_BATCH_SIZE",
    ),
    "ReciprocalRankFusionEnsembler": (
        "hashformers.ensemble.rrf_fusion",
        "ReciprocalRankFusionEnsembler",
    ),
    "Reranker": ("hashformers.beamsearch.reranker", "Reranker"),
    "Top2_Ensembler": (
        "hashformers.ensemble.top2_fusion",
        "Top2_Ensembler",
    ),
    "TransformerWordSegmenter": (
        "hashformers.segmenter.auto",
        "TransformerWordSegmenter",
    ),
    "WordSegmenterOutput": (
        "hashformers.segmenter.data_structures",
        "WordSegmenterOutput",
    ),
    "enforce_prob_dict": (
        "hashformers.beamsearch.data_structures",
        "enforce_prob_dict",
    ),
}
_LAZY_MODULES = {
    "base_segmenter": "hashformers.segmenter.base_segmenter",
    "beamsearch": "hashformers.beamsearch",
    "data_structures": "hashformers.segmenter.data_structures",
    "ensemble": "hashformers.ensemble",
    "evaluation": "hashformers.evaluation",
    "experiments": "hashformers.experiments",
    "segmenter": "hashformers.segmenter",
}

__all__ = [*_LAZY_EXPORTS, *_LAZY_MODULES]


def __getattr__(name: str) -> Any:
    """Load one historical top-level export on first use."""
    if name in _LAZY_MODULES:
        value = import_module(_LAZY_MODULES[name])
        globals()[name] = value
        return value
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include unresolved lazy exports in interactive discovery."""
    return sorted({*globals(), *__all__})
