import asyncio
import json
from unittest.mock import create_autospec, patch

import pandas as pd
import pytest

Client = pytest.importorskip("mcp").Client

import hashformers.mcp_server as mcp_server
from hashformers.segmenter.data_structures import WordSegmenterOutput
from hashformers.mcp_server import get_segmenter, main, mcp, segment_hashtags


@pytest.fixture(autouse=True)
def clear_segmenter_cache():
    """Clear the process-wide segmenter between tests.

    Yields:
        Control to the test with an empty segmenter cache.
    """
    mcp_server._segmenter = None
    yield
    mcp_server._segmenter = None


def _mock_output() -> WordSegmenterOutput:
    """Build a model-free segmentation result.

    Returns:
        A result containing two inputs and ranked candidates.
    """
    return WordSegmenterOutput(
        output=["ice cold", "benfica memes"],
        segmenter_rank=pd.DataFrame(
            [
                {
                    "characters": "icecold",
                    "segmentation": "icecold",
                    "score": 3.0,
                },
                {
                    "characters": "icecold",
                    "segmentation": "i ce cold",
                    "score": 2.0,
                },
                {
                    "characters": "icecold",
                    "segmentation": "ice cold",
                    "score": 1.0,
                },
                {
                    "characters": "benficamemes",
                    "segmentation": "benfica memes",
                    "score": 4.0,
                },
            ]
        ),
    )


def test_segment_hashtags_returns_serializable_ranked_candidates():
    segmenter = create_autospec(
        mcp_server.TransformerWordSegmenter,
        instance=True,
    )
    segmenter.segment.return_value = _mock_output()

    with (
        patch(
            "hashformers.mcp_server.TransformerWordSegmenter",
            return_value=segmenter,
        ) as segmenter_class,
        patch("hashformers.mcp_server.torch.cuda.is_available", return_value=False),
    ):
        result = segment_hashtags(["#icecold", "#benficamemes"], top_k=2)
        segment_hashtags(["#icecold"], top_k=1)

    assert result == {
        "results": [
            {
                "input": "#icecold",
                "selected_segmentation": "ice cold",
                "candidates": [
                    {"segmentation": "ice cold", "score": 1.0},
                    {"segmentation": "i ce cold", "score": 2.0},
                ],
            },
            {
                "input": "#benficamemes",
                "selected_segmentation": "benfica memes",
                "candidates": [
                    {"segmentation": "benfica memes", "score": 4.0},
                ],
            },
        ]
    }
    json.dumps(result, allow_nan=False)
    segmenter_class.assert_called_once_with(segmenter_device="cpu")
    segmenter.segment.assert_any_call(
        ["#icecold", "#benficamemes"],
        topk=2,
        return_ranks=True,
    )


def test_mcp_server_exposes_structured_segment_hashtags_tool():
    segmenter = create_autospec(
        mcp_server.TransformerWordSegmenter,
        instance=True,
    )
    segmenter.segment.return_value = _mock_output()

    async def call_tool():
        async with Client(mcp) as client:
            tools = await client.list_tools()
            result = await client.call_tool(
                "segment_hashtags",
                {"hashtags": ["#icecold"], "top_k": 1},
            )
            return tools, result

    with patch(
        "hashformers.mcp_server.TransformerWordSegmenter",
        return_value=segmenter,
    ):
        tools, result = asyncio.run(call_tool())

    assert [tool.name for tool in tools.tools] == ["segment_hashtags"]
    assert tools.tools[0].input_schema["properties"]["top_k"]["default"] == 5
    assert result.is_error is False
    assert result.structured_content == {
        "results": [
            {
                "input": "#icecold",
                "selected_segmentation": "ice cold",
                "candidates": [
                    {"segmentation": "ice cold", "score": 1.0},
                ],
            }
        ]
    }


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_segment_hashtags_rejects_invalid_top_k_without_loading_model(top_k):
    with patch("hashformers.mcp_server.TransformerWordSegmenter") as segmenter_class:
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            segment_hashtags(["#icecold"], top_k=top_k)

    segmenter_class.assert_not_called()


@pytest.mark.parametrize("hashtag", ["", "#", "###", "#   "])
def test_segment_hashtags_rejects_blank_hashtags_without_loading_model(hashtag):
    with patch("hashformers.mcp_server.TransformerWordSegmenter") as segmenter_class:
        with pytest.raises(ValueError, match="hashtags must contain text"):
            segment_hashtags([hashtag])

    segmenter_class.assert_not_called()


def test_segment_hashtags_handles_empty_input_without_loading_model():
    with patch("hashformers.mcp_server.TransformerWordSegmenter") as segmenter_class:
        assert segment_hashtags([]) == {"results": []}

    segmenter_class.assert_not_called()


def test_main_runs_server_over_stdio():
    with patch.object(mcp, "run") as run:
        main()

    run.assert_called_once_with(transport="stdio")
