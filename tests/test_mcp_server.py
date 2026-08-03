import asyncio
import json
import sqlite3
import stat
from pathlib import Path
from threading import Event, get_ident
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

Client = pytest.importorskip("mcp").Client

import hashformers.mcp_server as mcp_server
from hashformers.mcp_server import (
    ServerConfig,
    configure_server,
    continue_hashtag_file_job,
    get_segmenter,
    main,
    mcp,
    parse_server_config,
    rank_candidates,
    segment_hashtags,
    start_hashtag_file_job,
)
from hashformers.segmenter.data_structures import WordSegmenterOutput

requires_secure_file_jobs = pytest.mark.skipif(
    not mcp_server.SUPPORTS_SECURE_FILE_JOBS,
    reason="requires Linux descriptor-backed file operations",
)


@pytest.fixture(autouse=True)
def reset_server_state(tmp_path):
    """Reset process-wide MCP state between tests.

    Yields:
        Control to the test with default configuration and no loaded model.
    """
    configure_server(ServerConfig(file_roots=(str(tmp_path),)))
    yield
    configure_server(ServerConfig())


def _rank(rows):
    """Build a ranking DataFrame.

    Args:
        rows: ``segmentation``, ``score`` pairs for ``icecold``.

    Returns:
        A Hashformers-compatible ranking table.
    """
    return pd.DataFrame(
        [
            {
                "characters": segmentation.replace(" ", ""),
                "segmentation": segmentation,
                "score": score,
            }
            for segmentation, score in rows
        ]
    )


def _mock_output(include_pipeline=False):
    """Build a model-free ranked segmentation result.

    Args:
        include_pipeline: Whether to include reranker and ensemble ranks.

    Returns:
        Ranked output for one ``icecold`` input.
    """
    segmenter_rank = _rank([("ice cold", 1.0), ("i ce cold", 2.0), ("icecold", 3.0)])
    if not include_pipeline:
        return WordSegmenterOutput(
            output=["ice cold"],
            segmenter_rank=segmenter_rank,
        )
    return WordSegmenterOutput(
        output=["ice cold"],
        segmenter_rank=segmenter_rank,
        reranker_rank=_rank([("i ce cold", 0.5), ("ice cold", 1.5), ("icecold", 2.5)]),
        ensemble_rank=_rank([("ice cold", 0.0), ("i ce cold", 1.0)]),
    )


def _mock_batch_output(_segmenter, inputs, **kwargs):
    """Return each normalized input as its only segmentation.

    Args:
        _segmenter: Unused configured Transformer segmenter.
        inputs: Hashtags supplied by an MCP tool.
        **kwargs: Transformer pipeline options.

    Returns:
        Deterministic ranked output for the batch.
    """
    preprocessing = kwargs["preprocessing_kwargs"]
    normalized = mcp_server._normalize_inputs(
        inputs,
        lower=preprocessing["lower"],
        remove_hashtag=preprocessing["remove_hashtag"],
        hashtag_character=preprocessing["hashtag_character"],
    )
    rank = pd.DataFrame(
        [
            {
                "characters": value,
                "segmentation": value,
                "score": 1.0,
            }
            for value in normalized
        ]
    )
    return WordSegmenterOutput(output=normalized, segmenter_rank=rank)


def test_get_segmenter_forwards_complete_startup_configuration_once():
    """Verify model identity and memory options configure the singleton.

    """
    configure_server(
        ServerConfig(
            segmenter_model="custom/gpt",
            segmenter_model_type="incremental",
            segmenter_device="cpu",
            segmenter_batch_size=11,
            reranker_model="custom/bert",
            reranker_model_type="masked",
            reranker_device="inherit",
            reranker_batch_size=7,
        )
    )

    with patch("hashformers.mcp_server.TransformerWordSegmenter") as segmenter_class:
        first = get_segmenter()
        second = get_segmenter()

    assert first is second
    segmenter_class.assert_called_once_with(
        segmenter_model_name_or_path="custom/gpt",
        segmenter_model_type="incremental",
        segmenter_device="cpu",
        segmenter_gpu_batch_size=11,
        reranker_model_name_or_path="custom/bert",
        reranker_model_type="masked",
        reranker_device="cpu",
        reranker_gpu_batch_size=7,
    )


def test_get_segmenter_resolves_auto_device_for_both_models():
    """Verify auto and inherited devices resolve before model loading.

    """
    configure_server(ServerConfig(reranker_model="custom/bert"))

    with (
        patch("hashformers.mcp_server.torch.cuda.is_available", return_value=True),
        patch("hashformers.mcp_server.TransformerWordSegmenter") as segmenter_class,
    ):
        get_segmenter()

    assert segmenter_class.call_args.kwargs["segmenter_device"] == "cuda"
    assert segmenter_class.call_args.kwargs["reranker_device"] == "cuda"


def test_parse_server_config_accepts_every_model_option():
    """Verify every constructor-time capability is exposed by the CLI.

    """
    config = parse_server_config(
        [
            "--model",
            "custom/gpt",
            "--segmenter-model-type",
            "incremental",
            "--device",
            "cpu",
            "--batch-size",
            "12",
            "--reranker-model",
            "custom/bert",
            "--reranker-model-type",
            "masked",
            "--reranker-device",
            "cuda",
            "--reranker-batch-size",
            "8",
        ]
    )

    assert config == ServerConfig(
        segmenter_model="custom/gpt",
        segmenter_model_type="incremental",
        segmenter_device="cpu",
        segmenter_batch_size=12,
        reranker_model="custom/bert",
        reranker_model_type="masked",
        reranker_device="cuda",
        reranker_batch_size=8,
    )


def test_parse_server_config_rejects_invalid_batch_size():
    """Verify batch sizes cannot disable batching or become negative.

    """
    with pytest.raises(SystemExit):
        parse_server_config(["--batch-size", "0"])


def test_parse_server_config_accepts_file_policy(tmp_path):
    """Verify file roots and destructive overwrite require startup policy.

    """
    config = parse_server_config(
        [
            "--file-root",
            str(tmp_path),
            "--allow-file-overwrite",
        ]
    )

    assert config.file_roots == (str(tmp_path),)
    assert config.allow_file_overwrite is True


def test_run_transformer_pipeline_delegates_every_option_to_base_segmenter():
    """Verify the compatibility helper forwards every library option exactly.

    """
    segmenter = mcp_server.BaseWordSegmenter()
    segmenter_run = object()
    preprocessing_kwargs = {
        "lower": True,
        "remove_hashtag": False,
        "hashtag_character": "!",
    }
    expected = _mock_output()

    with patch.object(
        mcp_server.BaseWordSegmenter,
        "segment",
        return_value=expected,
    ) as base_segment:
        result = mcp_server._run_transformer_pipeline(
            segmenter,
            ["!IceCold"],
            top_k=20,
            steps=9,
            alpha=0.4,
            beta=0.6,
            strategy="reranker",
            preprocessing_kwargs=preprocessing_kwargs,
            segmenter_run=segmenter_run,
        )

    assert result is expected
    base_segment.assert_called_once_with(
        segmenter,
        ["!IceCold"],
        segmenter_run=segmenter_run,
        preprocessing_kwargs=preprocessing_kwargs,
        segmenter_kwargs={"topk": 20, "steps": 9},
        ensembler_kwargs={"alpha": 0.4, "beta": 0.6},
        use_reranker=True,
        use_ensembler=False,
        return_ranks=True,
    )


def test_segment_hashtags_separates_beam_width_from_response_limit():
    """Verify asking for fewer results does not narrow beam search.

    """
    segmenter = mcp_server.BaseWordSegmenter()
    pipeline = Mock(return_value=_mock_output())

    with (
        patch("hashformers.mcp_server.get_segmenter", return_value=segmenter),
        patch("hashformers.mcp_server._run_transformer_pipeline", pipeline),
    ):
        result = segment_hashtags(
            ["#IceCold"],
            top_k=20,
            steps=9,
            ranking_strategy="segmenter",
            lower=True,
            max_candidates=2,
        )

    assert result == {
        "results": [
            {
                "input": "#IceCold",
                "normalized_input": "icecold",
                "selected_segmentation": "ice cold",
                "ranking_strategy": "segmenter",
                "candidates": [
                    {"segmentation": "ice cold", "score": 1.0, "rank": 1},
                    {"segmentation": "i ce cold", "score": 2.0, "rank": 2},
                ],
                "component_rankings": None,
            }
        ]
    }
    pipeline.assert_called_once_with(
        segmenter,
        ["#IceCold"],
        top_k=20,
        steps=9,
        alpha=0.222,
        beta=0.111,
        strategy="segmenter",
        preprocessing_kwargs={
            "lower": True,
            "remove_hashtag": True,
            "hashtag_character": "#",
        },
    )
    json.dumps(result, allow_nan=False)


def test_segment_hashtags_returns_all_component_rankings():
    """Verify callers can inspect segmenter, reranker, and ensemble output.

    """
    configure_server(ServerConfig(reranker_model="custom/bert"))
    segmenter = mcp_server.BaseWordSegmenter()
    pipeline = Mock(return_value=_mock_output(include_pipeline=True))

    with (
        patch("hashformers.mcp_server.get_segmenter", return_value=segmenter),
        patch("hashformers.mcp_server._run_transformer_pipeline", pipeline),
    ):
        result = segment_hashtags(
            ["#icecold"],
            ranking_strategy="ensemble",
            max_candidates=64,
            include_component_rankings=True,
        )

    item = result["results"][0]
    assert item["ranking_strategy"] == "ensemble"
    assert [candidate["segmentation"] for candidate in item["candidates"]] == [
        "ice cold",
        "i ce cold",
    ]
    assert len(item["component_rankings"]["segmenter"]) == 3
    assert len(item["component_rankings"]["reranker"]) == 3
    assert len(item["component_rankings"]["ensemble"]) == 2
    pipeline.assert_called_once_with(
        segmenter,
        ["#icecold"],
        top_k=5,
        steps=5,
        alpha=0.222,
        beta=0.111,
        strategy="ensemble",
        preprocessing_kwargs={
            "lower": False,
            "remove_hashtag": True,
            "hashtag_character": "#",
        },
    )


def test_auto_strategy_uses_reranker_only_when_configured():
    """Verify auto preserves the Python pipeline's default selection.

    """
    segmenter = mcp_server.BaseWordSegmenter()
    pipeline = Mock(return_value=_mock_output())

    with (
        patch("hashformers.mcp_server.get_segmenter", return_value=segmenter),
        patch("hashformers.mcp_server._run_transformer_pipeline", pipeline),
    ):
        result = segment_hashtags(["#icecold"])

    assert result["results"][0]["ranking_strategy"] == "segmenter"
    assert pipeline.call_args.kwargs["strategy"] == "segmenter"

    configure_server(ServerConfig(reranker_model="custom/bert"))
    pipeline.reset_mock()
    pipeline.return_value = _mock_output(include_pipeline=True)
    with (
        patch("hashformers.mcp_server.get_segmenter", return_value=segmenter),
        patch("hashformers.mcp_server._run_transformer_pipeline", pipeline),
    ):
        result = segment_hashtags(["#icecold"])

    assert result["results"][0]["ranking_strategy"] == "ensemble"
    assert pipeline.call_args.kwargs["strategy"] == "ensemble"


@pytest.mark.parametrize("ranking_strategy", ["reranker", "ensemble"])
def test_reranker_strategies_require_a_startup_model(ranking_strategy):
    """Verify an unavailable pipeline component fails before model loading.

    """
    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        with pytest.raises(ValueError, match="requires --reranker-model"):
            segment_hashtags(
                ["#icecold"],
                ranking_strategy=ranking_strategy,
            )

    get_model.assert_not_called()


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("top_k", 0, "top_k must be a positive integer"),
        ("top_k", 65, "top_k must be at most 64"),
        ("steps", True, "steps must be a positive integer"),
        ("steps", 33, "steps must be at most 32"),
        ("max_candidates", -1, "max_candidates must be a positive integer"),
        ("max_candidates", 65, "max_candidates must be at most 64"),
        ("max_candidates", None, "max_candidates must be a positive integer"),
        ("alpha", float("inf"), "alpha must be a finite number"),
        ("hashtag_character", "##", "exactly one character"),
    ],
)
def test_segment_hashtags_rejects_invalid_options_before_loading(
    argument,
    value,
    message,
):
    """Verify invalid inference options cannot initialize a model.

    """
    options = {argument: value}
    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        with pytest.raises(ValueError, match=message):
            segment_hashtags(["#icecold"], **options)

    get_model.assert_not_called()


def test_segment_hashtags_rejects_oversized_interactive_batches():
    """Verify large datasets are routed to bounded file-job calls.

    """
    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        with pytest.raises(ValueError, match="at most 64 items"):
            segment_hashtags(["#tag"] * 65)

    get_model.assert_not_called()


def test_segment_hashtags_rejects_excessive_aggregate_beam_work():
    """Verify individually valid options cannot combine into an OOM-scale run.

    """
    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        with pytest.raises(ValueError, match="beam-search request is too large"):
            segment_hashtags(["a" * 512], top_k=64, steps=32)

    get_model.assert_not_called()


@pytest.mark.parametrize("hashtag", ["", "#", "###", "#   "])
def test_segment_hashtags_rejects_blank_normalized_inputs(hashtag):
    """Verify preprocessing cannot leave a content-free model input.

    """
    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        with pytest.raises(ValueError, match="after preprocessing"):
            segment_hashtags([hashtag])

    get_model.assert_not_called()


def test_segment_hashtags_empty_input_does_not_load_model():
    """Verify an empty batch returns immediately.

    """
    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        assert segment_hashtags([]) == {"results": []}

    get_model.assert_not_called()


@requires_secure_file_jobs
def test_hashtag_file_job_resumes_deduplicates_and_returns_only_summary(tmp_path):
    """Verify bulk input is checkpointed and repeated work is avoided.

    """
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text(
        "#icecold\n#benfica\n#icecold\n#mouraria\n#benfica\n",
        encoding="utf-8",
    )
    segmenter = mcp_server.BaseWordSegmenter()
    pipeline = Mock(side_effect=_mock_batch_output)
    context = Mock()
    context.report_progress = AsyncMock()

    started = start_hashtag_file_job(
        str(input_path),
        ranking_strategy="segmenter",
    )
    assert started["status"] == "in_progress"
    assert started["processed_unique"] == 0
    assert started["remaining_unique"] == 3
    assert not Path(started["output_path"]).exists()

    with (
        patch("hashformers.mcp_server.get_segmenter", return_value=segmenter),
        patch("hashformers.mcp_server._run_transformer_pipeline", pipeline),
    ):
        first = asyncio.run(
            continue_hashtag_file_job(
                started["job_path"],
                max_unique_hashtags=2,
                context=context,
            )
        )
        summary = asyncio.run(
            continue_hashtag_file_job(
                started["job_path"],
                max_unique_hashtags=2,
                context=context,
            )
        )

    assert first["status"] == "in_progress"
    assert first["processed_this_call"] == 2
    assert summary == {
        "job_path": str(
            input_path.with_name(
                "hashtags.txt.hashformers.jsonl.job.sqlite3"
            ).absolute()
        ),
        "input_path": str(input_path.absolute()),
        "output_path": str(
            input_path.with_name("hashtags.txt.hashformers.jsonl").absolute()
        ),
        "input_format": "text",
        "status": "completed",
        "total_hashtags": 5,
        "unique_hashtags": 3,
        "deduplicated_hashtags": 2,
        "processed_unique": 3,
        "processed_this_call": 1,
        "remaining_unique": 0,
        "segmenter_model": "gpt2",
        "reranker_model": None,
        "ranking_strategy": "segmenter",
    }
    assert [call.args[1] for call in pipeline.call_args_list] == [
        ["#icecold", "#benfica"],
        ["#mouraria"],
    ]
    assert [call.args[0] for call in context.report_progress.await_args_list] == [
        0,
        2,
        2,
        3,
    ]
    output_records = [
        json.loads(line)
        for line in Path(summary["output_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["source_line"] for record in output_records] == [1, 2, 3, 4, 5]
    assert [record["input"] for record in output_records] == [
        "#icecold",
        "#benfica",
        "#icecold",
        "#mouraria",
        "#benfica",
    ]

    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        repeated = asyncio.run(continue_hashtag_file_job(summary["job_path"]))
    assert repeated["status"] == "completed"
    assert repeated["processed_this_call"] == 0
    get_model.assert_not_called()


@requires_secure_file_jobs
def test_start_hashtag_file_job_fails_atomically_with_source_line(tmp_path):
    """Verify malformed bulk input leaves no partial destination.

    """
    input_path = tmp_path / "hashtags.txt"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text("#icecold\n\n#mouraria\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank hashtag at line 2"):
        start_hashtag_file_job(
            str(input_path),
            output_path=str(output_path),
            ranking_strategy="segmenter",
        )

    assert not output_path.exists()
    assert list(tmp_path.glob(".*.tmp")) == []
    assert list(tmp_path.glob("*.job.sqlite3")) == []


@requires_secure_file_jobs
def test_start_hashtag_file_job_rejects_oversized_physical_records(tmp_path):
    """Verify file parsing never allocates an unbounded source line.

    The failed start must leave neither a checkpoint nor a partial output.
    """
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text(
        "a" * (mcp_server.MAX_FILE_RECORD_CHARS + 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="file records must contain at most"):
        start_hashtag_file_job(
            str(input_path),
            ranking_strategy="segmenter",
        )

    assert list(tmp_path.glob("*.job.sqlite3")) == []


@requires_secure_file_jobs
def test_start_hashtag_file_job_rejects_unprocessable_beam_options(tmp_path):
    """Verify a published job always has a processable one-item chunk.

    Immutable search options must not leave a checkpoint permanently stuck.
    """
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text("a" * 512, encoding="utf-8")

    with pytest.raises(ValueError, match="beam-search request is too large"):
        start_hashtag_file_job(
            str(input_path),
            ranking_strategy="segmenter",
            top_k=64,
            steps=32,
        )

    assert list(tmp_path.glob("*.job.sqlite3")) == []


def test_start_hashtag_file_job_refuses_accidental_overwrite(tmp_path):
    """Verify bulk output cannot replace a file without explicit permission.

    """
    input_path = tmp_path / "hashtags.txt"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text("#icecold\n", encoding="utf-8")
    output_path.write_text("keep me", encoding="utf-8")

    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        with pytest.raises(ValueError, match="overwrite=true"):
            start_hashtag_file_job(
                str(input_path),
                output_path=str(output_path),
                ranking_strategy="segmenter",
            )

    assert output_path.read_text(encoding="utf-8") == "keep me"
    get_model.assert_not_called()


def test_start_hashtag_file_job_requires_startup_overwrite_policy(tmp_path):
    """Verify a tool argument cannot independently authorize replacement.

    """
    input_path = tmp_path / "hashtags.txt"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text("#icecold\n", encoding="utf-8")
    output_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="requires --allow-file-overwrite"):
        start_hashtag_file_job(
            str(input_path),
            output_path=str(output_path),
            overwrite=True,
            ranking_strategy="segmenter",
        )

    assert output_path.read_text(encoding="utf-8") == "keep me"


def test_start_hashtag_file_job_rejects_paths_outside_configured_roots(tmp_path):
    """Verify file tools cannot escape the operator-authorized directories.

    """
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_input = tmp_path / "outside.txt"
    outside_input.write_text("#icecold\n", encoding="utf-8")
    configure_server(ServerConfig(file_roots=(str(allowed_root),)))

    with pytest.raises(ValueError, match="outside configured file roots"):
        start_hashtag_file_job(
            str(outside_input),
            ranking_strategy="segmenter",
        )


@pytest.mark.parametrize(
    ("target_exists", "overwrite"),
    [(False, False), (True, True)],
)
def test_start_hashtag_file_job_authorizes_derived_output_symlink(
    tmp_path,
    target_exists,
    overwrite,
):
    """Verify a derived destination cannot follow a symlink outside its root."""
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    input_path = allowed_root / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")
    outside_output = tmp_path / "outside.jsonl"
    if target_exists:
        outside_output.write_text("keep me", encoding="utf-8")
    derived_output = allowed_root / "hashtags.txt.hashformers.jsonl"
    derived_output.symlink_to(outside_output)
    configure_server(
        ServerConfig(file_roots=(str(allowed_root),), allow_file_overwrite=True)
    )

    with pytest.raises(ValueError, match="outside configured file roots"):
        start_hashtag_file_job(
            str(input_path),
            overwrite=overwrite,
            ranking_strategy="segmenter",
        )

    assert not list(allowed_root.glob("*.job.sqlite3"))
    if target_exists:
        assert outside_output.read_text(encoding="utf-8") == "keep me"
    else:
        assert not outside_output.exists()


@requires_secure_file_jobs
def test_file_job_rejects_replaced_configured_root(tmp_path):
    """Verify a same-path directory replacement does not inherit authority.

    """
    allowed_root = tmp_path / "allowed"
    original_root = tmp_path / "allowed-original"
    allowed_root.mkdir()
    configure_server(ServerConfig(file_roots=(str(allowed_root),)))
    allowed_root.rename(original_root)
    allowed_root.mkdir()
    input_path = allowed_root / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")

    with pytest.raises(ValueError, match="configured file root changed"):
        start_hashtag_file_job(
            str(input_path),
            ranking_strategy="segmenter",
        )

    assert list(allowed_root.glob("*.job.sqlite3")) == []


def test_file_job_rejects_retargeted_configured_root_symlink(tmp_path):
    """Verify replacing a configured root with a symlink cannot widen access.

    """
    allowed_root = tmp_path / "allowed"
    original_root = tmp_path / "allowed-original"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    input_path = outside_root / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")
    configure_server(ServerConfig(file_roots=(str(allowed_root),)))
    allowed_root.rename(original_root)
    allowed_root.symlink_to(outside_root, target_is_directory=True)

    with pytest.raises(ValueError, match="outside configured file roots"):
        start_hashtag_file_job(
            str(allowed_root / input_path.name),
            ranking_strategy="segmenter",
        )

    assert list(outside_root.glob("*.job.sqlite3")) == []


@requires_secure_file_jobs
def test_reconfiguring_server_authorizes_replaced_file_root(tmp_path):
    """Verify explicit reconfiguration refreshes the pinned root identity.

    """
    allowed_root = tmp_path / "allowed"
    original_root = tmp_path / "allowed-original"
    allowed_root.mkdir()
    configure_server(ServerConfig(file_roots=(str(allowed_root),)))
    allowed_root.rename(original_root)
    allowed_root.mkdir()
    input_path = allowed_root / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")
    configure_server(ServerConfig(file_roots=(str(allowed_root),)))

    job = start_hashtag_file_job(
        str(input_path),
        ranking_strategy="segmenter",
    )

    assert Path(job["job_path"]).is_file()


def test_file_jobs_fail_closed_without_descriptor_support(tmp_path):
    """Verify unsupported hosts never fall back to raceable path operations.

    """
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")

    with patch(
        "hashformers.mcp_server.SUPPORTS_SECURE_FILE_JOBS",
        False,
    ):
        with pytest.raises(ValueError, match="secure file jobs are not supported"):
            start_hashtag_file_job(
                str(input_path),
                ranking_strategy="segmenter",
            )

    assert list(tmp_path.glob("*.job.sqlite3")) == []


@pytest.mark.skipif(
    not hasattr(mcp_server.os, "mkfifo")
    or not mcp_server.SUPPORTS_SECURE_FILE_JOBS,
    reason="requires secure POSIX file operations",
)
def test_file_job_rejects_source_replaced_with_fifo(tmp_path):
    """Verify a FIFO swap cannot block source validation indefinitely.

    """
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")
    real_open_child = mcp_server._open_child
    swapped = False

    def replace_source_before_open(*args, **kwargs):
        """Replace the source and assert its open is nonblocking.

        Args:
            *args: Positional arguments for ``_open_child``.
            **kwargs: Keyword arguments for ``_open_child``.

        Returns:
            A descriptor returned by the real helper.
        """
        nonlocal swapped
        if not swapped and args[2] == input_path.name:
            assert args[3] & mcp_server.os.O_NONBLOCK
            input_path.unlink()
            mcp_server.os.mkfifo(input_path)
            swapped = True
        return real_open_child(*args, **kwargs)

    with patch(
        "hashformers.mcp_server._open_child",
        side_effect=replace_source_before_open,
    ):
        with pytest.raises(ValueError, match="input_path is not a readable file"):
            start_hashtag_file_job(
                str(input_path),
                ranking_strategy="segmenter",
            )

    assert list(tmp_path.glob("*.job.sqlite3")) == []


def test_continue_hashtag_file_job_enforces_hard_batch_ceiling():
    """Verify callers cannot bypass the server's inference memory bound.

    """
    with pytest.raises(ValueError, match="max_unique_hashtags must be at most 64"):
        asyncio.run(
            continue_hashtag_file_job(
                "unused.job.sqlite3",
                max_unique_hashtags=65,
            )
        )


@requires_secure_file_jobs
def test_continue_hashtag_file_job_rejects_unmarked_sqlite_files(tmp_path):
    """Verify an arbitrary SQLite database cannot masquerade as a checkpoint.

    Validation happens before any Transformer model can be loaded.
    """
    checkpoint = tmp_path / "untrusted.sqlite3"
    with sqlite3.connect(checkpoint) as connection:
        connection.execute("CREATE TABLE metadata(key TEXT, value TEXT)")

    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        with pytest.raises(ValueError, match="not a valid Hashformers checkpoint"):
            asyncio.run(continue_hashtag_file_job(str(checkpoint)))

    get_model.assert_not_called()


@requires_secure_file_jobs
def test_continue_hashtag_file_job_uses_checkpoint_after_source_changes(tmp_path):
    """Verify continuation uses indexed records instead of rereading the source.

    """
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")
    job = start_hashtag_file_job(
        str(input_path),
        ranking_strategy="segmenter",
    )
    input_path.write_text("#mouraria\n", encoding="utf-8")

    pipeline = Mock(side_effect=_mock_batch_output)
    with (
        patch(
            "hashformers.mcp_server.get_segmenter",
            return_value=mcp_server.BaseWordSegmenter(),
        ),
        patch("hashformers.mcp_server._run_transformer_pipeline", pipeline),
    ):
        completed = asyncio.run(continue_hashtag_file_job(job["job_path"]))

    assert completed["status"] == "completed"
    assert pipeline.call_args.args[1] == ["#icecold"]


@requires_secure_file_jobs
def test_continue_hashtag_file_job_reconciles_completed_output(tmp_path):
    """Verify retry recovers from a crash after output rename but before flagging.

    """
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")
    job = start_hashtag_file_job(
        str(input_path),
        ranking_strategy="segmenter",
    )
    with (
        patch(
            "hashformers.mcp_server.get_segmenter",
            return_value=mcp_server.BaseWordSegmenter(),
        ),
        patch(
            "hashformers.mcp_server._run_transformer_pipeline",
            side_effect=_mock_batch_output,
        ),
    ):
        completed = asyncio.run(continue_hashtag_file_job(job["job_path"]))

    output_path = Path(completed["output_path"])
    expected_output = output_path.read_bytes()
    with sqlite3.connect(completed["job_path"]) as connection:
        connection.execute(
            "UPDATE metadata SET value = 'false' WHERE key = 'finalized'"
        )

    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        retried = asyncio.run(continue_hashtag_file_job(completed["job_path"]))

    assert retried["status"] == "completed"
    assert output_path.read_bytes() == expected_output
    get_model.assert_not_called()


@requires_secure_file_jobs
def test_file_job_never_overwrites_a_destination_created_during_publish(tmp_path):
    """Verify no-overwrite publication is atomic against a competing writer.

    """
    input_path = tmp_path / "hashtags.txt"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text("#icecold\n", encoding="utf-8")
    job = start_hashtag_file_job(
        str(input_path),
        output_path=str(output_path),
        ranking_strategy="segmenter",
    )
    real_link = mcp_server.os.link

    def create_competing_output(source, destination, *args, **kwargs):
        """Create the destination immediately before atomic publication.

        Args:
            source: Temporary output path.
            destination: Intended final output path.
            *args: Positional arguments forwarded to ``os.link``.
            **kwargs: Keyword arguments forwarded to ``os.link``.

        Returns:
            The result of the real hard-link operation.
        """
        descriptor = mcp_server.os.open(
            destination,
            mcp_server.os.O_WRONLY
            | mcp_server.os.O_CREAT
            | mcp_server.os.O_EXCL,
            0o600,
            dir_fd=kwargs["dst_dir_fd"],
        )
        with mcp_server.os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write("keep me")
        return real_link(source, destination, *args, **kwargs)

    with (
        patch(
            "hashformers.mcp_server.get_segmenter",
            return_value=mcp_server.BaseWordSegmenter(),
        ),
        patch(
            "hashformers.mcp_server._run_transformer_pipeline",
            side_effect=_mock_batch_output,
        ),
        patch("hashformers.mcp_server.os.link", side_effect=create_competing_output),
    ):
        with pytest.raises(ValueError, match="different content"):
            asyncio.run(continue_hashtag_file_job(job["job_path"]))

    assert output_path.read_text(encoding="utf-8") == "keep me"


@pytest.mark.skipif(
    not hasattr(mcp_server.os, "mkfifo")
    or not mcp_server.SUPPORTS_SECURE_FILE_JOBS,
    reason="requires secure POSIX file operations",
)
def test_file_job_rejects_fifo_created_during_publication(tmp_path):
    """Verify an empty competing FIFO is never accepted as valid output.

    """
    input_path = tmp_path / "hashtags.txt"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text("", encoding="utf-8")
    real_link = mcp_server.os.link
    real_open_child = mcp_server._open_child
    competing_output_created = False

    def create_competing_fifo(source, destination, *args, **kwargs):
        """Create a FIFO immediately before final output publication.

        Args:
            source: Temporary output basename.
            destination: Intended final output basename.
            *args: Positional arguments forwarded to ``os.link``.
            **kwargs: Descriptor arguments forwarded to ``os.link``.

        Returns:
            The result of the real hard-link operation.
        """
        nonlocal competing_output_created
        if destination == output_path.name:
            mcp_server.os.mkfifo(
                destination,
                dir_fd=kwargs["dst_dir_fd"],
            )
            competing_output_created = True
        return real_link(source, destination, *args, **kwargs)

    def assert_nonblocking_output_open(*args, **kwargs):
        """Require nonblocking flags when the competing output is inspected.

        Args:
            *args: Positional arguments for ``_open_child``.
            **kwargs: Keyword arguments for ``_open_child``.

        Returns:
            A descriptor returned by the real helper.
        """
        if competing_output_created and args[2] == output_path.name:
            assert args[3] & mcp_server.os.O_NONBLOCK
        return real_open_child(*args, **kwargs)

    with (
        patch(
            "hashformers.mcp_server.os.link",
            side_effect=create_competing_fifo,
        ),
        patch(
            "hashformers.mcp_server._open_child",
            side_effect=assert_nonblocking_output_open,
        ),
    ):
        with pytest.raises(ValueError, match="output_path must be a regular file"):
            start_hashtag_file_job(
                str(input_path),
                output_path=str(output_path),
                ranking_strategy="segmenter",
            )

    assert stat.S_ISFIFO(output_path.stat().st_mode)


@requires_secure_file_jobs
def test_file_job_publication_stays_bound_to_authorized_directory(tmp_path):
    """Verify a symlink swap cannot redirect final output outside a file root.

    Publication uses the directory descriptor opened before the swap.
    """
    allowed_root = tmp_path / "allowed"
    work_directory = allowed_root / "work"
    outside_directory = tmp_path / "outside"
    work_directory.mkdir(parents=True)
    outside_directory.mkdir()
    input_path = work_directory / "hashtags.txt"
    output_path = work_directory / "results.jsonl"
    input_path.write_text("#icecold\n", encoding="utf-8")
    configure_server(ServerConfig(file_roots=(str(allowed_root),)))
    job = start_hashtag_file_job(
        str(input_path),
        output_path=str(output_path),
        ranking_strategy="segmenter",
    )
    original_directory = allowed_root / "work-original"
    real_link = mcp_server.os.link
    swapped = False

    def swap_parent_then_link(source, destination, *args, **kwargs):
        """Swap the visible path, then publish through the anchored descriptor.

        Args:
            source: Temporary output basename.
            destination: Final output basename.
            *args: Positional arguments forwarded to ``os.link``.
            **kwargs: Descriptor arguments forwarded to ``os.link``.

        Returns:
            The result of the real hard-link operation.
        """
        nonlocal swapped
        if not swapped:
            work_directory.rename(original_directory)
            work_directory.symlink_to(outside_directory, target_is_directory=True)
            swapped = True
        return real_link(source, destination, *args, **kwargs)

    with (
        patch(
            "hashformers.mcp_server.get_segmenter",
            return_value=mcp_server.BaseWordSegmenter(),
        ),
        patch(
            "hashformers.mcp_server._run_transformer_pipeline",
            side_effect=_mock_batch_output,
        ),
        patch(
            "hashformers.mcp_server.os.link",
            side_effect=swap_parent_then_link,
        ),
    ):
        with pytest.raises(ValueError, match="directory changed"):
            asyncio.run(continue_hashtag_file_job(job["job_path"]))

    assert (original_directory / "results.jsonl").is_file()
    assert list(outside_directory.iterdir()) == []


@requires_secure_file_jobs
def test_concurrent_continuations_do_not_duplicate_inference(tmp_path):
    """Verify one checkpoint permits only one active inference claimant.

    """
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")
    job = start_hashtag_file_job(
        str(input_path),
        ranking_strategy="segmenter",
    )
    entered = Event()
    release = Event()

    def blocking_pipeline(segmenter, inputs, **kwargs):
        """Hold one model call while a competing continuation starts.

        Args:
            segmenter: Configured model wrapper.
            inputs: Current hashtag batch.
            **kwargs: Pipeline options.

        Returns:
            A deterministic ranked batch after the test releases it.
        """
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test did not release model inference")
        return _mock_batch_output(segmenter, inputs, **kwargs)

    async def exercise_concurrency():
        first_task = asyncio.create_task(
            continue_hashtag_file_job(job["job_path"])
        )
        while not entered.is_set():
            await asyncio.sleep(0.01)
        try:
            with pytest.raises(ValueError, match="already being processed"):
                await continue_hashtag_file_job(job["job_path"])
        finally:
            release.set()
        return await first_task

    pipeline = Mock(side_effect=blocking_pipeline)
    with (
        patch(
            "hashformers.mcp_server.get_segmenter",
            return_value=mcp_server.BaseWordSegmenter(),
        ),
        patch("hashformers.mcp_server._run_transformer_pipeline", pipeline),
    ):
        completed = asyncio.run(exercise_concurrency())

    assert completed["status"] == "completed"
    pipeline.assert_called_once()


@requires_secure_file_jobs
def test_cancelled_continuation_keeps_its_checkpoint_claim(tmp_path):
    """Verify cancellation cannot launch duplicate inference for one batch.

    """
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")
    job = start_hashtag_file_job(
        str(input_path),
        ranking_strategy="segmenter",
    )
    entered = Event()
    release = Event()

    def blocking_pipeline(segmenter, inputs, **kwargs):
        """Hold inference after cancellation until the test permits completion.

        Args:
            segmenter: Configured model wrapper.
            inputs: Current hashtag batch.
            **kwargs: Pipeline options.

        Returns:
            A deterministic ranked batch.
        """
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test did not release model inference")
        return _mock_batch_output(segmenter, inputs, **kwargs)

    async def exercise_cancellation():
        first_task = asyncio.create_task(
            continue_hashtag_file_job(job["job_path"])
        )
        while not entered.is_set():
            await asyncio.sleep(0.01)
        first_task.cancel()
        await asyncio.sleep(0.05)
        try:
            with pytest.raises(ValueError, match="already being processed"):
                await asyncio.wait_for(
                    continue_hashtag_file_job(job["job_path"]),
                    timeout=1,
                )
        finally:
            release.set()
        for _ in range(100):
            await asyncio.sleep(0.01)
            with sqlite3.connect(job["job_path"]) as connection:
                processed = connection.execute(
                    "SELECT COUNT(*) FROM unique_hashtags "
                    "WHERE result_json IS NOT NULL"
                ).fetchone()[0]
                finalized = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'finalized'"
                ).fetchone()[0]
            if processed == 1 and finalized == "true":
                break
        else:
            raise AssertionError("cancelled worker did not checkpoint its result")
        try:
            return await first_task
        except asyncio.CancelledError:
            return None

    pipeline = Mock(side_effect=blocking_pipeline)
    with (
        patch(
            "hashformers.mcp_server.get_segmenter",
            return_value=mcp_server.BaseWordSegmenter(),
        ),
        patch("hashformers.mcp_server._run_transformer_pipeline", pipeline),
    ):
        asyncio.run(exercise_cancellation())

    pipeline.assert_called_once()
    with sqlite3.connect(job["job_path"]) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM unique_hashtags WHERE result_json IS NOT NULL"
        ).fetchone()[0] == 1


@requires_secure_file_jobs
def test_large_file_finalization_runs_outside_the_event_loop(tmp_path):
    """Verify rendering a completed output does not block other MCP requests.

    """
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")
    job = start_hashtag_file_job(
        str(input_path),
        ranking_strategy="segmenter",
    )
    event_loop_thread = get_ident()
    finalizer_threads = []
    real_finalizer = mcp_server._finalize_file_job

    def record_finalizer_thread(*args, **kwargs):
        """Record the worker thread before delegating to the real finalizer.

        Args:
            *args: Positional finalizer arguments.
            **kwargs: Keyword finalizer arguments.

        Returns:
            The real finalizer result.
        """
        finalizer_threads.append(get_ident())
        return real_finalizer(*args, **kwargs)

    with (
        patch(
            "hashformers.mcp_server.get_segmenter",
            return_value=mcp_server.BaseWordSegmenter(),
        ),
        patch(
            "hashformers.mcp_server._run_transformer_pipeline",
            side_effect=_mock_batch_output,
        ),
        patch(
            "hashformers.mcp_server._finalize_file_job",
            side_effect=record_finalizer_thread,
        ),
    ):
        asyncio.run(continue_hashtag_file_job(job["job_path"]))

    assert finalizer_threads
    assert finalizer_threads[0] != event_loop_thread


@requires_secure_file_jobs
def test_empty_hashtag_file_job_completes_without_loading_model(tmp_path):
    """Verify empty input creates an empty final output immediately.

    """
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text("", encoding="utf-8")

    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        job = start_hashtag_file_job(
            str(input_path),
            ranking_strategy="segmenter",
        )

    assert job["status"] == "completed"
    assert job["total_hashtags"] == 0
    assert Path(job["output_path"]).read_text(encoding="utf-8") == ""
    get_model.assert_not_called()


@pytest.mark.parametrize(
    ("filename", "contents", "input_format", "input_field", "expected"),
    [
        (
            "hashtags.csv",
            "tag,label\n#icecold,a\n#mouraria,b\n",
            "csv",
            "tag",
            [(2, "#icecold"), (3, "#mouraria")],
        ),
        (
            "hashtags.jsonl",
            '"#icecold"\n{"tag": "#mouraria"}\n',
            "jsonl",
            "tag",
            [(1, "#icecold"), (2, "#mouraria")],
        ),
    ],
)
def test_file_reader_preserves_source_lines_across_formats(
    tmp_path,
    filename,
    contents,
    input_format,
    input_field,
    expected,
):
    """Verify CSV and JSON Lines inputs retain traceable source locations.

    """
    input_path = tmp_path / filename
    input_path.write_text(contents, encoding="utf-8")

    records = list(
        mcp_server._iter_file_hashtags(input_path, input_format, input_field)
    )

    assert records == expected


def test_csv_file_reader_preserves_physical_lines_for_multiline_records(tmp_path):
    """Verify CSV source locations count embedded newlines physically."""
    input_path = tmp_path / "hashtags.csv"
    input_path.write_text(
        'tag,label\n\n"#ice\ncold",a\n\n#mouraria,b\n',
        encoding="utf-8",
    )

    records = list(mcp_server._iter_file_hashtags(input_path, "csv", "tag"))

    assert records == [(3, "#ice\ncold"), (6, "#mouraria")]


def test_csv_file_reader_reports_physical_line_after_blank_rows(tmp_path):
    """Verify CSV validation errors identify a record's physical start."""
    input_path = tmp_path / "hashtags.csv"
    input_path.write_text(
        "tag,label\n\n#icecold,a\n\n,b\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="blank hashtag at CSV line 5"):
        list(mcp_server._iter_file_hashtags(input_path, "csv", "tag"))


def test_rank_candidates_selects_precomputed_scores_without_loading_model():
    """Verify direct selection never reruns beam search or loads a model.

    """
    candidate_sets = [
        {
            "input": "#icecold",
            "candidates": [
                {"segmentation": "ice cold", "score": 1.0},
                {"segmentation": "i ce cold", "score": 2.0},
            ],
        }
    ]

    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        result = rank_candidates(
            candidate_sets,
            ranking_strategy="segmenter",
            include_component_rankings=True,
        )

    assert result["results"][0]["selected_segmentation"] == "ice cold"
    assert result["results"][0]["component_rankings"]["reranker"] is None
    get_model.assert_not_called()


def test_rank_candidates_preserves_selected_tie_order():
    """Verify serialized ties match ProbabilityDictionary selection order."""
    candidate_sets = [
        {
            "input": "icecold",
            "candidates": [
                {"segmentation": "ice cold", "score": 1.0},
                {"segmentation": "i ce cold", "score": 1.0},
            ],
        }
    ]

    result = rank_candidates(
        candidate_sets,
        ranking_strategy="segmenter",
        max_candidates=1,
    )

    item = result["results"][0]
    assert item["selected_segmentation"] == "ice cold"
    assert item["candidates"] == [
        {"segmentation": "ice cold", "score": 1.0, "rank": 1}
    ]


def test_rank_candidates_preserves_multi_input_tie_selections():
    """Verify every selected tie is promoted before response truncation."""
    result = rank_candidates(
        [
            {
                "input": "aaaa",
                "candidates": [
                    {"segmentation": "aa aa", "score": 1.0},
                    {"segmentation": "a aaa", "score": 1.0},
                ],
            },
            {
                "input": "bbbb",
                "candidates": [
                    {"segmentation": "bb bb", "score": 0.0},
                    {"segmentation": "b bbb", "score": 0.0},
                ],
            },
        ],
        ranking_strategy="segmenter",
        max_candidates=1,
        include_component_rankings=True,
    )

    for item in result["results"]:
        assert item["candidates"][0]["segmentation"] == item[
            "selected_segmentation"
        ]
        assert item["component_rankings"]["segmenter"] == item["candidates"]


def test_rank_candidates_passes_precomputed_run_to_reranker_pipeline():
    """Verify reranking consumes supplied candidates instead of beam search.

    """
    configure_server(ServerConfig(reranker_model="custom/bert"))
    segmenter = mcp_server.BaseWordSegmenter()
    pipeline = Mock(
        return_value=WordSegmenterOutput(
            output=["i ce cold"],
            segmenter_rank=_rank([("ice cold", 1.0), ("i ce cold", 2.0)]),
            reranker_rank=_rank([("i ce cold", 0.5), ("ice cold", 1.5)]),
        )
    )

    with (
        patch("hashformers.mcp_server.get_segmenter", return_value=segmenter),
        patch("hashformers.mcp_server._run_transformer_pipeline", pipeline),
    ):
        result = rank_candidates(
            [
                {
                    "input": "icecold",
                    "candidates": [
                        {"segmentation": "ice cold", "score": 1.0},
                        {"segmentation": "i ce cold", "score": 2.0},
                    ],
                }
            ],
            ranking_strategy="reranker",
        )

    assert result["results"][0]["selected_segmentation"] == "i ce cold"
    assert pipeline.call_args.args == (segmenter, ["icecold"])
    kwargs = pipeline.call_args.kwargs
    assert kwargs["segmenter_run"].dictionary == {
        "ice cold": 1.0,
        "i ce cold": 2.0,
    }
    assert kwargs["strategy"] == "reranker"


def test_rank_candidates_batches_candidate_sets_for_reranking():
    """Verify multiple candidate sets share one GPU-backed reranker call.

    """
    configure_server(ServerConfig(reranker_model="custom/bert"))

    def batched_output(_segmenter, inputs, **kwargs):
        """Return the supplied global precomputed ranking unchanged.

        Args:
            _segmenter: Unused configured model wrapper.
            inputs: Normalized candidate-set inputs.
            **kwargs: Pipeline options including the precomputed run.

        Returns:
            A ranked output aligned with every input.
        """
        rank = kwargs["segmenter_run"].to_dataframe()
        return WordSegmenterOutput(
            output=["ice cold", "ben fica"],
            segmenter_rank=rank,
            reranker_rank=rank,
        )

    pipeline = Mock(side_effect=batched_output)
    with (
        patch(
            "hashformers.mcp_server.get_segmenter",
            return_value=mcp_server.BaseWordSegmenter(),
        ),
        patch("hashformers.mcp_server._run_transformer_pipeline", pipeline),
    ):
        result = rank_candidates(
            [
                {
                    "input": "icecold",
                    "candidates": [
                        {"segmentation": "ice cold", "score": 1.0},
                        {"segmentation": "i ce cold", "score": 2.0},
                    ],
                },
                {
                    "input": "benfica",
                    "candidates": [
                        {"segmentation": "ben fica", "score": 1.0},
                        {"segmentation": "benfi ca", "score": 2.0},
                    ],
                },
            ],
            ranking_strategy="reranker",
        )

    assert [item["selected_segmentation"] for item in result["results"]] == [
        "ice cold",
        "ben fica",
    ]
    pipeline.assert_called_once()
    assert pipeline.call_args.args[1] == ["icecold", "benfica"]


def test_rank_candidates_isolates_colliding_candidate_sets():
    """Verify equal character strings do not share each other's hypotheses.

    """
    configure_server(ServerConfig(reranker_model="custom/bert"))

    def isolated_output(_segmenter, inputs, **kwargs):
        """Return only the candidates supplied in the current isolated batch.

        Args:
            _segmenter: Unused configured model wrapper.
            inputs: Normalized candidate-set inputs.
            **kwargs: Pipeline options including the precomputed run.

        Returns:
            A ranked output aligned with the isolated input.
        """
        rank = kwargs["segmenter_run"].to_dataframe()
        selected = rank.sort_values("score").iloc[0]["segmentation"]
        return WordSegmenterOutput(
            output=[selected],
            segmenter_rank=rank,
            reranker_rank=rank,
        )

    pipeline = Mock(side_effect=isolated_output)
    with (
        patch(
            "hashformers.mcp_server.get_segmenter",
            return_value=mcp_server.BaseWordSegmenter(),
        ),
        patch("hashformers.mcp_server._run_transformer_pipeline", pipeline),
    ):
        result = rank_candidates(
            [
                {
                    "input": "icecold",
                    "candidates": [
                        {"segmentation": "ice cold", "score": 1.0},
                    ],
                },
                {
                    "input": "icecold",
                    "candidates": [
                        {"segmentation": "i ce cold", "score": 1.0},
                    ],
                },
            ],
            ranking_strategy="reranker",
        )

    assert [item["selected_segmentation"] for item in result["results"]] == [
        "ice cold",
        "i ce cold",
    ]
    assert pipeline.call_count == 2


def test_rank_candidates_rejects_unbounded_whitespace_hypotheses():
    """Verify candidate boundaries cannot bypass request-size ceilings.

    """
    with pytest.raises(ValueError, match="single spaces as boundaries"):
        rank_candidates(
            [
                {
                    "input": "icecold",
                    "candidates": [
                        {"segmentation": "ice" + (" " * 5_000) + "cold", "score": 1},
                    ],
                }
            ],
            ranking_strategy="segmenter",
        )


@pytest.mark.parametrize(
    "candidate_sets",
    [
        [{"input": "icecold", "candidates": []}],
        [
            {
                "input": "icecold",
                "candidates": [
                    {"segmentation": "ice cold", "score": 1.0},
                    {"segmentation": "ice cold", "score": 2.0},
                ],
            }
        ],
        [
            {
                "input": "icecold",
                "candidates": [
                    {"segmentation": "wrong text", "score": 1.0},
                ],
            }
        ],
    ],
)
def test_rank_candidates_rejects_invalid_candidate_sets(candidate_sets):
    """Verify malformed precomputed runs are rejected before model loading.

    """
    with patch("hashformers.mcp_server.get_segmenter") as get_model:
        with pytest.raises(ValueError):
            rank_candidates(candidate_sets, ranking_strategy="segmenter")

    get_model.assert_not_called()


def test_mcp_server_exposes_complete_structured_surface():
    """Verify agents discover every supported MCP segmentation workflow.

    """

    async def list_tools():
        async with Client(mcp) as client:
            return await client.list_tools()

    tools = asyncio.run(list_tools()).tools
    tools_by_name = {tool.name: tool for tool in tools}

    assert set(tools_by_name) == {
        "segment_hashtags",
        "start_hashtag_file_job",
        "continue_hashtag_file_job",
        "rank_candidates",
    }
    assert tools_by_name["segment_hashtags"].annotations.read_only_hint is True
    assert tools_by_name["start_hashtag_file_job"].annotations.destructive_hint is True
    assert (
        tools_by_name["continue_hashtag_file_job"].annotations.open_world_hint
        is True
    )
    hashtag_schema = tools_by_name["segment_hashtags"].input_schema["properties"]
    assert hashtag_schema["top_k"]["default"] == 5
    assert hashtag_schema["steps"]["default"] == 5
    assert hashtag_schema["max_candidates"]["default"] == 5
    assert hashtag_schema["ranking_strategy"]["default"] == "auto"
    file_schema = tools_by_name["continue_hashtag_file_job"].input_schema["properties"]
    assert "context" not in file_schema


def test_mcp_transport_validates_nested_candidate_contract():
    """Verify precomputed candidates cross the real MCP serialization layer.

    """

    async def call_tool():
        async with Client(mcp) as client:
            return await client.call_tool(
                "rank_candidates",
                {
                    "candidate_sets": [
                        {
                            "input": "icecold",
                            "candidates": [
                                {"segmentation": "ice cold", "score": 1.0},
                                {"segmentation": "i ce cold", "score": 2.0},
                            ],
                        }
                    ],
                    "ranking_strategy": "segmenter",
                },
            )

    result = asyncio.run(call_tool())

    assert result.is_error is False
    assert result.structured_content["results"][0]["selected_segmentation"] == (
        "ice cold"
    )


def test_main_configures_server_and_runs_stdio():
    """Verify the console entry point applies CLI settings before serving.

    """
    with patch.object(mcp, "run") as run:
        main(["--model", "custom/gpt", "--batch-size", "9"])

    assert mcp_server._server_config.segmenter_model == "custom/gpt"
    assert mcp_server._server_config.segmenter_batch_size == 9
    run.assert_called_once_with(transport="stdio")
