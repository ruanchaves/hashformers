"""Expose Hashformers hashtag segmentation through a local MCP server.

The module registers one structured tool and runs it over stdio for local MCP
clients.
"""

from threading import Lock

import torch
from mcp.server import MCPServer
from typing_extensions import TypedDict

from hashformers.segmenter.auto import TransformerWordSegmenter


class Candidate(TypedDict):
    """Represent a ranked segmentation candidate.

    Attributes:
        segmentation: Candidate text with inferred word boundaries.
        score: Language-model score; lower values rank higher.
    """

    segmentation: str
    score: float


class SegmentationResult(TypedDict):
    """Represent the segmentation result for one input hashtag.

    Attributes:
        input: Original hashtag supplied by the caller.
        selected_segmentation: Highest-ranked segmentation.
        candidates: Ranked segmentation alternatives.
    """

    input: str
    selected_segmentation: str
    candidates: list[Candidate]


class SegmentHashtagsResult(TypedDict):
    """Represent the structured result returned by ``segment_hashtags``.

    Attributes:
        results: Per-input segmentation results in input order.
    """

    results: list[SegmentationResult]


_segmenter: TransformerWordSegmenter | None = None
_segmenter_lock = Lock()


def get_segmenter() -> TransformerWordSegmenter:
    """Return the process-wide Transformer word segmenter.

    Returns:
        The segmenter reused by every MCP tool call in this process.
    """
    global _segmenter
    if _segmenter is None:
        with _segmenter_lock:
            if _segmenter is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _segmenter = TransformerWordSegmenter(segmenter_device=device)
    return _segmenter


def _ranked_candidates(
    rank,
    selected_segmentation: str,
    top_k: int,
) -> list[Candidate]:
    """Serialize the top candidates for one hashtag.

    Args:
        rank: Candidate ranking returned by Hashformers.
        selected_segmentation: Selected segmentation returned by Hashformers.
        top_k: Maximum number of candidates to return.

    Returns:
        JSON-serializable candidate records ordered by score.
    """
    characters = selected_segmentation.replace(" ", "")
    rows = rank[rank["characters"] == characters]
    rows = rows.sort_values("score").head(top_k)
    return [
        {
            "segmentation": str(row.segmentation),
            "score": float(row.score),
        }
        for row in rows.itertuples(index=False)
    ]


mcp = MCPServer(
    "hashformers",
    description="Segment hashtags into words with Transformer language models.",
)


@mcp.tool()
def segment_hashtags(
    hashtags: list[str],
    top_k: int = 5,
) -> SegmentHashtagsResult:
    """Segment hashtags and return their ranked candidates.

    Args:
        hashtags: Hashtags to segment.
        top_k: Maximum number of ranked candidates to return per hashtag.

    Returns:
        The selected segmentation and candidates ranked by ascending score for
        each input. Lower scores rank higher.

    Raises:
        ValueError: If ``top_k`` is not a positive integer or a hashtag has no
            text to segment.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if any(
        not isinstance(hashtag, str) or not hashtag.lstrip("#").strip()
        for hashtag in hashtags
    ):
        raise ValueError("hashtags must contain text to segment")
    if not hashtags:
        return {"results": []}

    output = get_segmenter().segment(
        hashtags,
        topk=top_k,
        return_ranks=True,
    )
    rank = output.ensemble_rank
    if rank is None:
        rank = output.reranker_rank
    if rank is None:
        rank = output.segmenter_rank

    return {
        "results": [
            {
                "input": hashtag,
                "selected_segmentation": str(segmentation),
                "candidates": _ranked_candidates(rank, str(segmentation), top_k),
            }
            for hashtag, segmentation in zip(hashtags, output.output)
        ]
    }


def main() -> None:
    """Run the Hashformers MCP server over stdio.

    Returns:
        None.
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
