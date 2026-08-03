import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest

import hashformers.mcp_server as mcp_server
from hashformers.mcp_server import (
    ServerConfig,
    configure_models,
    configure_server,
    continue_hashtag_file_job,
    discover_huggingface_models,
    get_segmenter,
    parse_server_config,
    sample_hashtag_file,
    segment_hashtags,
    start_hashtag_file_job,
)
from hashformers.segmenter.data_structures import WordSegmenterOutput


SEGMENTER_REVISION = "1" * 40
RERANKER_REVISION = "2" * 40
OTHER_REVISION = "3" * 40

requires_secure_file_jobs = pytest.mark.skipif(
    not mcp_server.SUPPORTS_SECURE_FILE_JOBS,
    reason="requires Linux descriptor-backed file operations",
)


@pytest.fixture(autouse=True)
def reset_server_state(tmp_path):
    """Give every model-selection test fresh process-wide MCP state."""
    configure_server(ServerConfig(file_roots=(str(tmp_path),)))
    yield
    configure_server(ServerConfig())


def _model_info(
    repository_id,
    revision,
    *,
    language="en",
    architecture="GPT2LMHeadModel",
    model_type=None,
    pipeline_tag="text-generation",
    parameters=100_000_000,
    download_size=400_000_000,
    gated=False,
    private=False,
    auto_map=None,
    downloads=100,
):
    """Build complete fake Hub metadata without making network requests."""
    if model_type is None:
        model_type = {
            "GPT2LMHeadModel": "gpt2",
            "BertForMaskedLM": "bert",
        }.get(architecture, "unknown-test-model")
    config = {"architectures": [architecture], "model_type": model_type}
    if auto_map is not None:
        config["auto_map"] = auto_map
    return SimpleNamespace(
        id=repository_id,
        sha=revision,
        private=private,
        gated=gated,
        disabled=False,
        library_name="transformers",
        tags=["transformers", language],
        card_data=SimpleNamespace(language=[language]),
        config=config,
        transformers_info=None,
        safetensors=SimpleNamespace(total=parameters),
        siblings=[
            SimpleNamespace(
                rfilename="model.safetensors",
                size=download_size,
                lfs=None,
            )
        ],
        used_storage=download_size,
        pipeline_tag=pipeline_tag,
        downloads=downloads,
        likes=10,
    )


def _configured_api(include_reranker=False):
    """Return a fake HfApi serving exact public model revisions."""
    api = Mock()
    infos = {
        "example/segmenter": _model_info(
            "example/segmenter",
            SEGMENTER_REVISION,
        ),
    }
    if include_reranker:
        infos["example/reranker"] = _model_info(
            "example/reranker",
            RERANKER_REVISION,
            architecture="BertForMaskedLM",
            pipeline_tag="fill-mask",
        )
    api.model_info.side_effect = lambda repository_id, **_kwargs: infos[
        repository_id
    ]
    return api


def _enable_deferred(tmp_path, **changes):
    """Start the in-process MCP state in explicit deferred-selection mode."""
    configure_server(
        ServerConfig(
            defer_model_selection=True,
            file_roots=(str(tmp_path),),
            **changes,
        )
    )


def test_download_size_matches_files_selected_for_pinned_snapshot():
    info = _model_info("example/model", SEGMENTER_REVISION, download_size=400)
    info.siblings.append(
        SimpleNamespace(
            rfilename="model.onnx",
            size=10_000,
            lfs=None,
        )
    )

    assert mcp_server._hub_download_size(info) == 400


def test_download_size_is_unknown_when_selected_file_metadata_is_incomplete():
    info = _model_info("example/model", SEGMENTER_REVISION)
    info.siblings[0].size = None

    assert mcp_server._hub_download_size(info) is None


def test_parse_server_config_accepts_deferred_selection_and_size_ceilings(tmp_path):
    """Verify the operator explicitly controls deferred selection and bounds."""
    config = parse_server_config(
        [
            "--defer-model-selection",
            "--max-model-parameters",
            "123",
            "--max-model-size-bytes",
            "456",
            "--file-root",
            str(tmp_path),
        ]
    )

    assert config.defer_model_selection is True
    assert config.max_model_parameters == 123
    assert config.max_model_size_bytes == 456


@requires_secure_file_jobs
def test_sample_hashtag_file_is_streaming_bounded_distinct_and_deterministic(
    tmp_path,
):
    """Verify a large duplicate-heavy file returns only a fixed reservoir."""
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text(
        "\n".join([f"#tag{index}" for index in range(100)] + ["#tag1"] * 50)
        + "\n",
        encoding="utf-8",
    )
    _enable_deferred(tmp_path)

    first = sample_hashtag_file(str(input_path))
    second = sample_hashtag_file(str(input_path))

    assert first == second
    assert first["total_hashtags"] == 150
    assert first["sample_size"] == mcp_server.MAX_FILE_SAMPLES
    assert len(first["samples"]) == len(set(first["samples"])) == 20
    assert first["input_format"] == "text"
    assert first["file_size_bytes"] == input_path.stat().st_size
    assert first["deferred_model_selection"] is True
    assert first["models_configured"] is False


@requires_secure_file_jobs
@pytest.mark.parametrize(
    ("filename", "contents", "input_format", "input_field", "expected"),
    [
        (
            "hashtags.csv",
            "tag,label\n#bomdia,pt\n#goodmorning,en\n",
            "csv",
            "tag",
            {"#bomdia", "#goodmorning"},
        ),
        (
            "hashtags.jsonl",
            '{"tag":"#доброеутро"}\n{"tag":"#おはよう"}\n',
            "jsonl",
            "tag",
            {"#доброеутро", "#おはよう"},
        ),
    ],
)
def test_sample_hashtag_file_supports_structured_and_mixed_language_inputs(
    tmp_path,
    filename,
    contents,
    input_format,
    input_field,
    expected,
):
    """Keep mixed samples intact so the agent can report ambiguity itself."""
    input_path = tmp_path / filename
    input_path.write_text(contents, encoding="utf-8")

    result = sample_hashtag_file(
        str(input_path),
        input_format=input_format,
        input_field=input_field,
    )

    assert set(result["samples"]) == expected
    assert result["total_hashtags"] == 2
    assert result["input_format"] == input_format


def test_sample_hashtag_file_enforces_hard_limit_before_opening_file(tmp_path):
    """Verify callers cannot expand the sample beyond the context-safe cap."""
    with pytest.raises(ValueError, match="sample_size must be at most 20"):
        sample_hashtag_file(
            str(tmp_path / "missing.txt"),
            sample_size=21,
        )


@pytest.mark.parametrize("language", ["en", "pt"])
def test_discover_huggingface_models_validates_languages_and_exact_revisions(
    tmp_path,
    language,
):
    """Cover English and non-English deterministic Hub discovery."""
    _enable_deferred(tmp_path)
    revision = SEGMENTER_REVISION if language == "en" else OTHER_REVISION
    repository_id = f"example/{language}-model"
    summary = SimpleNamespace(
        id=repository_id,
        sha=revision,
        downloads=200,
    )
    api = Mock()
    api.list_models.return_value = [summary]
    api.model_info.return_value = _model_info(
        repository_id,
        revision,
        language=language,
    )

    with patch("hashformers.mcp_server.HfApi", return_value=api):
        result = discover_huggingface_models(language, "segmenter")

    assert result["language"] == language
    assert result["models_configured"] is False
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["repository_id"] == repository_id
    assert candidate["revision"] == revision
    assert candidate["scorer_type"] == "gpt2"
    assert language in candidate["language_tags"]
    assert "public non-gated" in candidate["reason"]
    expected_list_options = dict(
        filter=("transformers", language),
        gated=False,
        sort="downloads",
        limit=mcp_server.MAX_MODEL_DISCOVERY_SCAN,
        full=True,
        cardData=True,
        fetch_config=True,
        token=False,
    )
    if mcp_server.HUB_SUPPORTS_PARAMETER_FILTER:
        expected_list_options["num_parameters"] = (
            f"max:{mcp_server.DEFAULT_MAX_MODEL_PARAMETERS}"
        )
    api.list_models.assert_called_once_with(**expected_list_options)


def test_discovery_skips_unavailable_gated_oversize_custom_and_unloadable_models(
    tmp_path,
):
    """Verify every unsafe candidate is excluded without configuring state."""
    _enable_deferred(
        tmp_path,
        max_model_parameters=50,
        max_model_size_bytes=500,
    )
    repository_ids = [
        "x/unavailable",
        "x/gated",
        "x/large",
        "x/custom",
        "x/unsupported-weights",
    ]
    summaries = [
        SimpleNamespace(id=name, sha=str(index + 4) * 40, downloads=10 - index)
        for index, name in enumerate(repository_ids)
    ]
    api = Mock()
    api.list_models.return_value = summaries

    def model_info(repository_id, **_kwargs):
        if repository_id == "x/unavailable":
            raise OSError("not found")
        if repository_id == "x/gated":
            return _model_info(repository_id, "5" * 40, gated="manual")
        if repository_id == "x/large":
            return _model_info(
                repository_id,
                "6" * 40,
                parameters=51,
                download_size=100,
            )
        if repository_id == "x/custom":
            return _model_info(
                repository_id,
                "7" * 40,
                parameters=10,
                download_size=100,
                auto_map={"AutoModel": "model.CustomModel"},
            )
        info = _model_info(
            repository_id,
            "8" * 40,
            parameters=10,
            download_size=100,
        )
        info.siblings[0].rfilename = "model.onnx"
        return info

    api.model_info.side_effect = model_info
    with patch("hashformers.mcp_server.HfApi", return_value=api):
        result = discover_huggingface_models("en", "segmenter")

    assert result["candidates"] == []
    assert mcp_server._models_configured() is False
    assert mcp_server._server_config.segmenter_model is None


def test_failed_discovery_leaves_deferred_server_unconfigured(tmp_path):
    """Keep remote search failures atomic and free of implicit fallbacks."""
    _enable_deferred(tmp_path)
    api = Mock()
    api.list_models.side_effect = OSError("Hub unavailable")

    with patch("hashformers.mcp_server.HfApi", return_value=api):
        with pytest.raises(ValueError, match="model discovery failed"):
            discover_huggingface_models("en", "segmenter")

    assert mcp_server._models_configured() is False
    assert mcp_server._validated_model_selection is None


def test_discovery_is_deterministic_and_hard_caps_results(tmp_path):
    """Use popularity then repository ID for a stable bounded shortlist."""
    _enable_deferred(tmp_path)
    summaries = [
        SimpleNamespace(id="x/z", sha="4" * 40, downloads=10),
        SimpleNamespace(id="x/a", sha="5" * 40, downloads=10),
        SimpleNamespace(id="x/b", sha="6" * 40, downloads=20),
    ]
    infos = {
        summary.id: _model_info(
            summary.id,
            summary.sha,
            downloads=summary.downloads,
        )
        for summary in summaries
    }
    api = Mock()
    api.list_models.return_value = summaries
    api.model_info.side_effect = lambda repository_id, **_kwargs: infos[
        repository_id
    ]

    with patch("hashformers.mcp_server.HfApi", return_value=api):
        result = discover_huggingface_models("en", "segmenter", limit=2)

    assert [candidate["repository_id"] for candidate in result["candidates"]] == [
        "x/b",
        "x/a",
    ]
    with pytest.raises(ValueError, match="limit must be at most 10"):
        discover_huggingface_models("en", "segmenter", limit=11)


def test_configure_models_is_validated_lazy_idempotent_and_immutable(tmp_path):
    """Publish one exact selection without downloading or constructing models."""
    _enable_deferred(tmp_path)
    api = _configured_api(include_reranker=True)
    with (
        patch("hashformers.mcp_server.HfApi", return_value=api),
        patch("hashformers.mcp_server.snapshot_download") as download,
        patch("hashformers.mcp_server.TransformerWordSegmenter") as constructor,
    ):
        first = configure_models(
            "example/segmenter",
            SEGMENTER_REVISION,
            reranker_model="example/reranker",
            reranker_revision=RERANKER_REVISION,
        )
        repeated = configure_models(
            "example/segmenter",
            SEGMENTER_REVISION,
            reranker_model="example/reranker",
            reranker_revision=RERANKER_REVISION,
        )
        with pytest.raises(ValueError, match="restart the MCP server"):
            configure_models(
                "example/segmenter",
                OTHER_REVISION,
                reranker_model="example/reranker",
                reranker_revision=RERANKER_REVISION,
            )

    assert first == repeated
    assert first["models"]["segmenter"]["revision"] == SEGMENTER_REVISION
    assert first["models"]["reranker"]["revision"] == RERANKER_REVISION
    assert api.model_info.call_count == 2
    download.assert_not_called()
    constructor.assert_not_called()


def test_configuration_failure_does_not_publish_a_partial_selection(tmp_path):
    """Validate both roles before changing process-wide model configuration."""
    _enable_deferred(tmp_path)
    api = _configured_api(include_reranker=True)
    reranker = _model_info(
        "example/reranker",
        RERANKER_REVISION,
        architecture="BertForMaskedLM",
        pipeline_tag="fill-mask",
        gated="manual",
    )
    api.model_info.side_effect = [
        _model_info("example/segmenter", SEGMENTER_REVISION),
        reranker,
    ]

    with patch("hashformers.mcp_server.HfApi", return_value=api):
        with pytest.raises(ValueError, match="gated Hub models"):
            configure_models(
                "example/segmenter",
                SEGMENTER_REVISION,
                reranker_model="example/reranker",
                reranker_revision=RERANKER_REVISION,
            )

    assert mcp_server._models_configured() is False
    assert mcp_server._validated_model_selection is None
    assert mcp_server._server_config.segmenter_model is None
    with pytest.raises(ValueError, match="model selection is deferred"):
        segment_hashtags(["#icecold"])


@pytest.mark.parametrize(
    ("info", "message"),
    [
        (
            _model_info(
                "example/segmenter",
                SEGMENTER_REVISION,
                architecture="FutureForCausalLM",
                model_type="future",
            ),
            "not registered by the installed Transformers",
        ),
        (
            _model_info("example/segmenter", SEGMENTER_REVISION),
            "incomplete size metadata",
        ),
    ],
)
def test_configuration_fails_closed_for_unsupported_or_unbounded_models(
    tmp_path,
    info,
    message,
):
    """Reject selections that cannot load locally or enforce the byte ceiling."""
    _enable_deferred(tmp_path)
    if "incomplete" in message:
        info.siblings[0].size = None
    api = Mock()
    api.model_info.return_value = info

    with patch("hashformers.mcp_server.HfApi", return_value=api):
        with pytest.raises(ValueError, match=message):
            configure_models("example/segmenter", SEGMENTER_REVISION)

    assert mcp_server._models_configured() is False
    assert mcp_server._server_config.segmenter_model is None


def test_concurrent_identical_configuration_validates_and_publishes_once(tmp_path):
    """Serialize concurrent configuration calls around one atomic validation."""
    _enable_deferred(tmp_path)
    api = _configured_api()
    with patch("hashformers.mcp_server.HfApi", return_value=api):
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda _index: configure_models(
                        "example/segmenter",
                        SEGMENTER_REVISION,
                    ),
                    range(4),
                )
            )

    assert all(result == results[0] for result in results)
    api.model_info.assert_called_once()


def test_concurrent_lazy_loading_retains_one_pinned_segmenter_instance(tmp_path):
    """Ensure concurrent inference cannot create multiple resident models."""
    _enable_deferred(tmp_path)
    api = _configured_api()
    with patch("hashformers.mcp_server.HfApi", return_value=api):
        configure_models("example/segmenter", SEGMENTER_REVISION)

    resident = object()
    with (
        patch(
            "hashformers.mcp_server.snapshot_download",
            return_value="/cache/exact-segmenter",
        ) as download,
        patch(
            "hashformers.mcp_server.TransformerWordSegmenter",
            return_value=resident,
        ) as constructor,
    ):
        with ThreadPoolExecutor(max_workers=4) as executor:
            instances = list(executor.map(lambda _index: get_segmenter(), range(4)))

    assert instances == [resident] * 4
    download.assert_called_once_with(
        repo_id="example/segmenter",
        revision=SEGMENTER_REVISION,
        token=False,
        allow_patterns=mcp_server.HUB_MODEL_FILE_PATTERNS,
    )
    constructor.assert_called_once()
    assert constructor.call_args.kwargs["segmenter_model_name_or_path"] == (
        "/cache/exact-segmenter"
    )


def test_segmentation_response_records_selected_repository_and_revision(tmp_path):
    """Expose exact model identity with every interactive model result."""
    _enable_deferred(tmp_path)
    api = _configured_api()
    with patch("hashformers.mcp_server.HfApi", return_value=api):
        configure_models("example/segmenter", SEGMENTER_REVISION)

    rank = pd.DataFrame(
        [{"characters": "icecold", "segmentation": "ice cold", "score": 1.0}]
    )
    output = WordSegmenterOutput(output=["ice cold"], segmenter_rank=rank)
    with (
        patch("hashformers.mcp_server.get_segmenter", return_value=object()),
        patch(
            "hashformers.mcp_server._run_transformer_pipeline",
            return_value=output,
        ),
    ):
        result = segment_hashtags(["#icecold"], ranking_strategy="segmenter")

    assert result["models"]["segmenter"]["repository_id"] == (
        "example/segmenter"
    )
    assert result["models"]["segmenter"]["revision"] == SEGMENTER_REVISION


@requires_secure_file_jobs
def test_file_job_status_checkpoint_and_output_record_exact_revision(tmp_path):
    """Persist reproducibility metadata through the complete file workflow."""
    input_path = tmp_path / "hashtags.txt"
    input_path.write_text("#icecold\n", encoding="utf-8")
    _enable_deferred(tmp_path)
    api = _configured_api()
    with patch("hashformers.mcp_server.HfApi", return_value=api):
        configure_models("example/segmenter", SEGMENTER_REVISION)

    started = start_hashtag_file_job(
        str(input_path),
        ranking_strategy="segmenter",
    )
    model_result = {
        "results": [
            {
                "input": "#icecold",
                "normalized_input": "icecold",
                "selected_segmentation": "ice cold",
                "ranking_strategy": "segmenter",
                "candidates": [
                    {"segmentation": "ice cold", "score": 1.0, "rank": 1}
                ],
                "component_rankings": None,
            }
        ]
    }
    with patch("hashformers.mcp_server.segment_hashtags", return_value=model_result):
        status = asyncio.run(continue_hashtag_file_job(started["job_path"]))

    assert status["segmenter_model"] == "example/segmenter"
    assert status["segmenter_revision"] == SEGMENTER_REVISION
    assert status["reranker_revision"] is None
    record = json.loads(
        Path(status["output_path"]).read_text(encoding="utf-8")
    )
    assert record["models"]["segmenter"] == {
        "repository_id": "example/segmenter",
        "revision": SEGMENTER_REVISION,
        "scorer_type": "gpt2",
    }


def test_configure_models_requires_explicit_operator_authorization():
    """Do not turn ordinary startup into a caller-controlled download surface."""
    with pytest.raises(ValueError, match="requires --defer-model-selection"):
        configure_models("example/segmenter", SEGMENTER_REVISION)
