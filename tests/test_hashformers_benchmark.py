import json
import subprocess
import sys
from argparse import ArgumentTypeError
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.hashformers_benchmark import (
    EVALUATION_CONTRACT_ID,
    MODEL_SPECS,
    PROTOCOL_ID,
    candidate_batch_size,
    compare_runs,
    resolve_model_snapshot,
    segment_once,
    select_manifest,
    stable_id_hash,
)
from scripts.qwen_benchmark import SCHEMA_VERSION, load_jsonl, summarize_records

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_RESULTS = (
    REPOSITORY_ROOT / "benchmarks/qwen/results/2026-08-03-colab-t4-fp16-v3"
)
PUBLISHED_RUNNER_REVISION = "d4180e11e383608387685d8f595103adfae8ee72"


def prediction(sample_id, *, model, protocol, correct=False):
    source = f"{sample_id}value"
    gold = f"{sample_id} value"
    raw_output = gold if correct else source
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol,
        "evaluation_contract_id": EVALUATION_CONTRACT_ID,
        "manifest_sha256": "f" * 64,
        "sample_id": sample_id,
        "dataset": "example/data",
        "dataset_revision": "b" * 40,
        "split": "test",
        "row_index": ord(sample_id[0]),
        "group": "English Hashtags",
        "input": source,
        "gold": gold,
        "model_label": model,
        "model_id": f"example/{model}",
        "model_revision": "a" * 40,
        "requested_precision": "library-default",
        "actual_parameter_dtype": "torch.float32",
        "quantization": "none",
        "resolved_device": "cuda:0",
        "raw_output": raw_output,
        "output_wrapper": None,
        "prediction": raw_output,
        "prediction_source": "model_output",
        "recovery_method": None,
        "valid": True,
        "invalid_reason": None,
        "strict_correct": correct,
        "correct": correct,
        "generation_ms": 10.0,
        "error": None,
    }


def test_hashformers_model_pins_and_scopes_are_explicit():
    assert MODEL_SPECS == {
        "gpt2": {
            "model_id": "openai-community/gpt2",
            "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            "label": "Hashformers-GPT2",
            "scope_dataset": None,
            "status": "historical-hashformers-baseline-refreshed-protocol",
        },
        "distilgpt2": {
            "model_id": "distilbert/distilgpt2",
            "revision": "2290a62682d06624634c1f46a6ad5be0f47f38aa",
            "label": "Hashformers-DistilGPT2",
            "scope_dataset": None,
            "status": "historical-hashformers-baseline-refreshed-protocol",
        },
        "rugpt3small": {
            "model_id": "ai-forever/rugpt3small_based_on_gpt2",
            "revision": "a9307e696cd3c5b7f953ff4cb19d76a4d81821d5",
            "label": "Hashformers-RuGPT3Small",
            "scope_dataset": "ruanchaves/nru_hse",
            "status": "historical-russian-baseline-refreshed-protocol",
        },
    }


def test_candidate_batch_size_accepts_pr_80_auto_or_a_positive_integer():
    assert candidate_batch_size("auto") == "auto"
    assert candidate_batch_size("AUTO") == "auto"
    assert candidate_batch_size("64") == 64
    for value in ("0", "-1", "not-a-size"):
        with pytest.raises(ArgumentTypeError, match="positive integer or auto"):
            candidate_batch_size(value)


def test_runner_is_executable_directly_from_its_script_path():
    result = subprocess.run(
        [sys.executable, "scripts/hashformers_benchmark.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "fixed issue #78 samples" in result.stdout


def test_model_snapshot_must_resolve_to_requested_commit():
    revision = MODEL_SPECS["gpt2"]["revision"]

    def download_snapshot(**kwargs):
        assert kwargs == {
            "repo_id": "openai-community/gpt2",
            "revision": revision,
            "token": False,
        }
        return f"/cache/models--openai-community--gpt2/snapshots/{revision}"

    assert (
        resolve_model_snapshot(
            "openai-community/gpt2",
            revision,
            download_snapshot=download_snapshot,
        ).name
        == revision
    )

    with pytest.raises(RuntimeError, match="did not resolve"):
        resolve_model_snapshot(
            "openai-community/gpt2",
            revision,
            download_snapshot=lambda **_: "/cache/models--gpt2/main",
        )


def test_russian_model_scope_selects_only_pinned_nru_records():
    manifest = [
        {"sample_id": "one", "dataset": "ruanchaves/nru_hse"},
        {"sample_id": "two", "dataset": "ruanchaves/boun"},
    ]

    assert select_manifest(manifest, None) == manifest
    assert select_manifest(manifest, "ruanchaves/nru_hse") == [manifest[0]]
    with pytest.raises(ValueError, match="no records"):
        select_manifest(manifest, "missing/data")


def test_segment_once_synchronizes_and_preserves_one_result():
    calls = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda device: calls.append(("sync", device)))
    )

    class Segmenter:
        def segment(self, inputs, **kwargs):
            calls.append((inputs, kwargs))
            return ["ice cream"]

    output, elapsed_ms = segment_once(
        torch,
        Segmenter(),
        "icecream",
        device="cuda:0",
        topk=5,
        steps=5,
    )

    assert output == "ice cream"
    assert elapsed_ms >= 0
    assert calls == [
        ("sync", "cuda:0"),
        (["icecream"], {"topk": 5, "steps": 5, "use_reranker": False}),
        ("sync", "cuda:0"),
    ]


def test_cross_protocol_comparison_pairs_shared_ids_and_subset_runs():
    qwen = [
        prediction("a", model="Qwen", protocol="qwen-v3", correct=True),
        prediction("b", model="Qwen", protocol="qwen-v3", correct=False),
    ]
    gpt2 = [
        prediction("a", model="GPT2", protocol=PROTOCOL_ID, correct=False),
        prediction("b", model="GPT2", protocol=PROTOCOL_ID, correct=True),
    ]
    russian = [
        prediction("b", model="RuGPT", protocol=PROTOCOL_ID, correct=True),
    ]

    comparison = compare_runs([qwen, gpt2, russian])

    assert comparison["evaluation_contract_id"] == EVALUATION_CONTRACT_ID
    assert len(comparison["runs"]) == 3
    assert [pair["paired_samples"] for pair in comparison["paired_comparisons"]] == [
        2,
        1,
        1,
    ]
    assert comparison["paired_comparisons"][0]["accuracy_difference"] == 0.0
    assert comparison["paired_comparisons"][1]["accuracy_difference"] == -1.0
    assert comparison["paired_comparisons"][0]["left_protocol_id"] == "qwen-v3"
    assert comparison["paired_comparisons"][0]["right_protocol_id"] == PROTOCOL_ID
    assert stable_id_hash(["b", "a"]) == stable_id_hash(["a", "b"])


def test_cross_protocol_comparison_rejects_manifest_or_provenance_mismatch():
    left = [prediction("a", model="Qwen", protocol="qwen-v3")]
    right = [prediction("a", model="GPT2", protocol=PROTOCOL_ID)]
    right[0]["manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="same manifest hash"):
        compare_runs([left, right])

    right[0]["manifest_sha256"] = "f" * 64
    right[0]["gold"] = "av alue"
    with pytest.raises(ValueError, match="differs in gold"):
        compare_runs([left, right])


def test_published_adaptive_hashformers_results_are_clean_and_recomputable():
    runs = []
    for directory, model_key, expected_count in (
        ("hashformers-gpt2", "gpt2", 280),
        ("hashformers-distilgpt2", "distilgpt2", 280),
        ("hashformers-rugpt3small", "rugpt3small", 20),
    ):
        predictions = load_jsonl(PUBLISHED_RESULTS / directory / "predictions.jsonl")
        metadata = json.loads(
            (PUBLISHED_RESULTS / directory / "run_metadata.json").read_text(
                encoding="utf-8"
            )
        )

        assert len(predictions) == expected_count
        assert metadata["status"] == "completed"
        assert metadata["repository_revision"] == PUBLISHED_RUNNER_REVISION
        assert metadata["repository_dirty"] is False
        assert metadata["measurement"]["runtime_error_count"] == 0
        assert metadata["segmentation"]["gpu_batch_size"] == "auto"
        assert metadata["segmentation"]["max_gpu_batch_size"] == 512
        telemetry = metadata["measurement"]["candidate_batch_telemetry"]
        assert telemetry["configured_batch_size"] == "auto"
        assert telemetry["max_batch_size"] == 512
        assert telemetry["oom_backoff_events"] == 0
        assert metadata["runtime"]["model_commit"] == MODEL_SPECS[model_key]["revision"]
        assert metadata["results"] == summarize_records(predictions)
        assert all(
            record["prediction_source"] == "model_output" for record in predictions
        )
        runs.append(predictions)

    qwen_runs = [
        load_jsonl(PUBLISHED_RESULTS / model / "predictions.jsonl")
        for model in ("qwen3", "qwen2")
    ]
    published = json.loads(
        (PUBLISHED_RESULTS / "combined_comparison.json").read_text(encoding="utf-8")
    )
    recomputed = compare_runs([*qwen_runs, *runs])

    for field in (
        "schema_version",
        "evaluation_contract_id",
        "manifest_sha256",
        "runs",
        "paired_comparisons",
    ):
        assert published[field] == recomputed[field]
