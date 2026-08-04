import hashlib
import json
from argparse import ArgumentTypeError
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.build_qwen_sample_manifest import selected_indices
from scripts.qwen_benchmark import (
    MODEL_SPECS,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    cpu_metadata,
    generate_once,
    load_jsonl,
    paired_bootstrap_interval,
    paired_comparisons,
    parse_insertion_only,
    propose_segmentation,
    resolve_hub_file_commit,
    single_device,
    summarize_records,
    validate_insertion_only,
    validate_manifest,
    validate_revision,
    wilson_interval,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "benchmarks/qwen/samples.jsonl"
MANIFEST_SHA256 = "743e7519eb4ef760f45a7b5b6a34fea3b0f7394b85e9fed7609b27864cd8497d"
PUBLISHED_RESULTS = (
    REPOSITORY_ROOT / "benchmarks/qwen/results/2026-08-03-colab-t4-fp16-v3"
)


def load_manifest():
    return [
        json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    ]


def prediction(sample_id, *, model="left", valid=True, correct=False, group="English"):
    source = f"{sample_id}value"
    gold = f"{sample_id} value"
    raw_output = gold if correct else source if valid else "changed"
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": "f" * 64,
        "model_label": model,
        "model_id": f"example/{model}",
        "model_revision": "a" * 40,
        "requested_precision": "bfloat16",
        "actual_parameter_dtype": "torch.bfloat16",
        "quantization": "none",
        "resolved_device": "cuda:0",
        "dataset": "example/data",
        "dataset_revision": "b" * 40,
        "split": "test",
        "row_index": int(sample_id == "two"),
        "input": source,
        "gold": gold,
        "raw_output": raw_output,
        "output_wrapper": None,
        "prediction": raw_output if valid else source,
        "prediction_source": "model_output" if valid else "source_fallback",
        "recovery_method": None if valid else "unchanged_input",
        "valid": valid,
        "invalid_reason": None if valid else "changed_non_space_characters",
        "strict_correct": bool(valid and correct),
        "correct": correct,
        "generation_ms": 10.0,
        "group": group,
        "error": None,
    }


@pytest.mark.parametrize(
    ("source", "raw_output", "expected"),
    [
        ("icecream", "ice cream", (True, "ice cream", None)),
        ("icecream", "  ice  cream ", (True, "ice cream", None)),
        ("icecream", '"ice cream"', (True, "ice cream", None)),
        ("icecream", "'ice cream'", (True, "ice cream", None)),
        ("CamelCase", "Camel Case", (True, "Camel Case", None)),
        ("CamelCase", "camel Case", (False, None, "changed_non_space_characters")),
        ("icecream", "ice\ncream", (False, None, "changed_non_space_characters")),
        (
            "icecream",
            '\n"ice cream"\n',
            (False, None, "changed_non_space_characters"),
        ),
        (
            "icecream",
            "Result: ice cream",
            (False, None, "changed_non_space_characters"),
        ),
        ("icecream", "ice", (False, None, "changed_non_space_characters")),
        ("icecream", "", (False, None, "empty_output")),
        ("icecream", None, (False, None, "missing_output")),
    ],
)
def test_strict_output_contract_does_not_hide_character_changes(
    source, raw_output, expected
):
    assert validate_insertion_only(source, raw_output) == expected


def test_output_contract_records_only_a_matching_quote_envelope():
    assert parse_insertion_only("icecream", '"ice cream"') == (
        True,
        "ice cream",
        None,
        "matching_ascii_quotes",
    )
    assert parse_insertion_only("icecream", '"ice cream') == (
        False,
        None,
        "changed_non_space_characters",
        None,
    )
    assert parse_insertion_only("icecream", '"ice cold"') == (
        False,
        None,
        "changed_non_space_characters",
        None,
    )


@pytest.mark.parametrize(
    ("source", "raw_output", "expected_prediction", "expected_source", "method"),
    [
        (
            "CamelCase",
            "camel Case",
            "Camel Case",
            "recovered_model_output",
            "case_preserving_projection",
        ),
        (
            "PleaseDontSuck",
            "Please Do Not Suck",
            "Please Do nt Suck",
            "recovered_model_output",
            "edit_alignment_projection",
        ),
        (
            "drawingday",
            "drawing_day",
            "drawing day",
            "recovered_model_output",
            "case_preserving_projection",
        ),
        (
            "icecream",
            "Result: ice cream",
            "ice cream",
            "recovered_model_output",
            "case_preserving_projection",
        ),
        (
            "icecream",
            "```\nice cream\n```",
            "ice cream",
            "recovered_model_output",
            "case_preserving_projection",
        ),
        (
            "icecream",
            "unrelated",
            "icecream",
            "source_fallback",
            "unchanged_input",
        ),
        (
            "icecream",
            "",
            "icecream",
            "source_fallback",
            "unchanged_input",
        ),
    ],
)
def test_invalid_output_still_produces_an_auditable_segmentation_proposal(
    source, raw_output, expected_prediction, expected_source, method
):
    (
        valid,
        prediction_value,
        invalid_reason,
        output_wrapper,
        prediction_source,
        recovery_method,
    ) = propose_segmentation(source, raw_output)

    assert valid is False
    assert prediction_value == expected_prediction
    assert invalid_reason in {"changed_non_space_characters", "empty_output"}
    assert output_wrapper is None
    assert prediction_source == expected_source
    assert recovery_method == method
    assert prediction_value.replace(" ", "") == source


def test_valid_output_bypasses_recovery_and_preserves_wrapper_audit():
    assert propose_segmentation("icecream", '"ice cream"') == (
        True,
        "ice cream",
        None,
        "matching_ascii_quotes",
        "model_output",
        None,
    )


def test_manifest_is_fixed_complete_and_auditable():
    records = load_manifest()
    validate_manifest(records)

    assert len(records) == 280
    assert len({record["sample_id"] for record in records}) == 280
    assert {record["dataset"] for record in records} == {
        "ruanchaves/boun",
        "ruanchaves/stan_small",
        "ruanchaves/stan_large",
        "ruanchaves/dev_stanford",
        "ruanchaves/test_stanford",
        "ruanchaves/snap",
        "ruanchaves/nru_hse",
        "ruanchaves/hashset_distant",
        "ruanchaves/hashset_distant_sampled",
        "ruanchaves/loyola",
        "ruanchaves/lynx",
        "ruanchaves/jhotdraw",
        "ruanchaves/binkley",
        "ruanchaves/bt11",
    }
    assert all(len(record["dataset_revision"]) == 40 for record in records)
    assert all(
        record["gold"].replace(" ", "").casefold() == record["input"].casefold()
        for record in records
    )
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == MANIFEST_SHA256


def test_sample_selection_is_stable_and_dataset_local():
    assert selected_indices("example/data", 10, 4) == [1, 2, 6, 7]
    assert selected_indices("example/data", 10, 4) == selected_indices(
        "example/data", 10, 4
    )
    assert selected_indices("example/other", 10, 4) != selected_indices(
        "example/data", 10, 4
    )


def test_manifest_validation_rejects_duplicate_ids():
    records = load_manifest()[:2]
    records[1]["sample_id"] = records[0]["sample_id"]
    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_manifest(records)


def test_manifest_validation_enforces_insertion_only_gold():
    records = load_manifest()[:1]
    records[0]["gold"] += "!"

    with pytest.raises(ValueError, match="only by inserted spaces"):
        validate_manifest(records)


def test_model_revision_must_be_an_exact_commit_sha():
    assert validate_revision("a" * 40) == "a" * 40
    for revision in ("main", "A" * 40, "a" * 39, "a" * 41):
        with pytest.raises(ValueError, match="exact 40-character"):
            validate_revision(revision)


def test_tokenizer_revision_falls_back_to_hub_snapshot_path():
    revision = "a" * 40

    def download_file(**kwargs):
        assert kwargs == {
            "repo_id": "example/model",
            "filename": "tokenizer_config.json",
            "revision": revision,
            "token": False,
        }
        return (
            f"/cache/models--example--model/snapshots/{revision}/tokenizer_config.json"
        )

    assert (
        resolve_hub_file_commit(
            "example/model",
            revision,
            "tokenizer_config.json",
            download_file=download_file,
        )
        == revision
    )


def test_tokenizer_revision_rejects_unversioned_cache_path():
    with pytest.raises(RuntimeError, match="could not resolve an immutable Hub commit"):
        resolve_hub_file_commit(
            "example/model",
            "a" * 40,
            "tokenizer_config.json",
            download_file=lambda **_: "/cache/models--example--model/tokenizer.json",
        )


def test_benchmark_device_must_be_one_explicit_cpu_or_cuda_device():
    assert single_device("cpu") == "cpu"
    assert single_device("cuda") == "cuda:0"
    assert single_device("CUDA:2") == "cuda:2"
    with pytest.raises(ArgumentTypeError, match="device must be"):
        single_device("auto")


def test_generation_uses_one_chat_template_without_duplicate_special_tokens():
    class FakeBatch(dict):
        def to(self, device):
            self.device = device
            return self

    input_ids = SimpleNamespace(shape=(1, 2))
    batch = FakeBatch(input_ids=input_ids)
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    tokenizer.return_value = batch
    tokenizer.decode.return_value = "ice cold"
    model = Mock(device=None)
    model.generate.return_value = [[101, 102, 103]]
    torch = SimpleNamespace(inference_mode=lambda: nullcontext())

    raw_output, _, _, generated_tokens = generate_once(
        torch,
        tokenizer,
        model,
        MODEL_SPECS["qwen3"],
        "icecold",
        64,
    )

    tokenizer.apply_chat_template.assert_called_once_with(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(source="icecold"),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokenizer.assert_called_once_with(
        "formatted prompt",
        return_tensors="pt",
        add_special_tokens=False,
    )
    assert raw_output == "ice cold"
    assert generated_tokens == 1


def test_wilson_interval_and_paired_bootstrap_are_bounded_and_deterministic():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.236593, abs=1e-6)
    assert high == pytest.approx(0.763407, abs=1e-6)
    assert paired_bootstrap_interval([True] * 5, [False] * 5) == (1.0, 1.0)
    assert paired_bootstrap_interval([True, False], [True, False]) == (0.0, 0.0)


def test_summary_scores_fallbacks_but_reports_strict_validity_separately():
    records = [
        prediction("one", valid=True, correct=True),
        prediction("two", valid=False, correct=False),
    ]
    summary = summarize_records(records)["overall"]

    assert summary["accuracy"]["successes"] == 1
    assert summary["accuracy"]["total"] == 2
    assert summary["accuracy"]["rate"] == 0.5
    assert summary["invalid_output_rate"]["successes"] == 1
    assert summary["invalid_output_rate"]["rate"] == 0.5
    assert summary["strict_output_accuracy"]["successes"] == 1
    assert summary["source_fallback_rate"]["successes"] == 1
    assert summary["recovered_prediction_rate"]["successes"] == 0
    assert summary["runtime_error_rate"]["successes"] == 0
    assert summary["output_wrapper_rate"]["successes"] == 0
    assert summary["throughput_items_per_second"] == pytest.approx(100.0)


def test_summary_reports_accepted_output_wrappers_separately_from_invalid_outputs():
    record = prediction("one", valid=True, correct=True)
    record["raw_output"] = f'"{record["raw_output"]}"'
    record["output_wrapper"] = "matching_ascii_quotes"

    summary = summarize_records([record])["overall"]

    assert summary["accuracy"]["successes"] == 1
    assert summary["invalid_output_rate"]["successes"] == 0
    assert summary["output_wrapper_rate"]["successes"] == 1


def test_summary_scores_recovered_invalid_output_without_calling_it_valid():
    record = prediction("one", valid=False, correct=True)
    record.update(
        {
            "raw_output": "One value",
            "prediction": "one value",
            "prediction_source": "recovered_model_output",
            "recovery_method": "case_preserving_projection",
        }
    )

    summary = summarize_records([record])["overall"]

    assert summary["accuracy"]["successes"] == 1
    assert summary["strict_output_accuracy"]["successes"] == 0
    assert summary["invalid_output_rate"]["successes"] == 1
    assert summary["recovered_prediction_rate"]["successes"] == 1
    assert summary["source_fallback_rate"]["successes"] == 0


def test_summary_separates_runtime_errors_from_invalid_model_outputs():
    record = prediction("one", valid=False, correct=False)
    record["error"] = "RuntimeError: backend failed"
    record["generation_ms"] = None
    record["raw_output"] = None
    record["prediction"] = None
    record["prediction_source"] = None
    record["recovery_method"] = None
    record["invalid_reason"] = "runtime_error"

    summary = summarize_records([record])["overall"]

    assert summary["invalid_output_rate"]["successes"] == 0
    assert summary["runtime_error_rate"]["successes"] == 1


def test_summary_rejects_missing_or_inconsistent_prediction_fields():
    missing = prediction("one", valid=True, correct=True)
    del missing["raw_output"]
    with pytest.raises(ValueError, match="is missing: raw_output"):
        summarize_records([missing])

    inconsistent = prediction("one", valid=True, correct=True)
    inconsistent["correct"] = False
    with pytest.raises(ValueError, match="correct does not match"):
        summarize_records([inconsistent])

    inconsistent_source = prediction("one", valid=False, correct=False)
    inconsistent_source["prediction_source"] = "model_output"
    with pytest.raises(ValueError, match="inconsistent segmentation proposal"):
        summarize_records([inconsistent_source])

    unsupported_wrapper = prediction("one", valid=True, correct=True)
    unsupported_wrapper["output_wrapper"] = "code_fence"
    with pytest.raises(ValueError, match="unsupported output_wrapper"):
        summarize_records([unsupported_wrapper])


def test_paired_comparison_joins_by_stable_sample_id():
    left = [
        prediction("one", model="left", correct=True),
        prediction("two", model="left", correct=False),
    ]
    right = [
        prediction("two", model="right", correct=False),
        prediction("one", model="right", correct=False),
    ]

    comparison = paired_comparisons([left, right])[0]
    assert comparison["paired_samples"] == 2
    assert comparison["accuracy_difference"] == 0.5
    assert comparison["left"] == "left"
    assert comparison["right"] == "right"
    assert comparison["protocol_id"] == PROTOCOL_ID
    assert comparison["manifest_sha256"] == "f" * 64
    assert comparison["left_configuration"]["model_id"] == "example/left"
    assert comparison["right_configuration"]["model_id"] == "example/right"


def test_paired_comparison_rejects_different_samples_or_protocols():
    left = [prediction("one", model="left")]
    with pytest.raises(ValueError, match="same sample IDs"):
        paired_comparisons([left, [prediction("two", model="right")]])

    right = [prediction("one", model="right")]
    right[0]["protocol_id"] = "different-protocol"
    with pytest.raises(ValueError, match="same protocol_id"):
        paired_comparisons([left, right])


def test_model_pins_preserve_qwen2_and_disable_qwen3_thinking():
    assert MODEL_SPECS["qwen3"] == {
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "label": "Qwen3-0.6B (text-only, non-thinking)",
        "enable_thinking": False,
        "status": "current-fallback",
    }
    assert MODEL_SPECS["qwen2-historical"]["model_id"] == "Qwen/Qwen2-0.5B-Instruct"
    assert MODEL_SPECS["qwen2-historical"]["status"] == (
        "historical-model-under-refreshed-protocol"
    )


def test_published_gpu_results_are_complete_clean_and_recomputable():
    manifest_ids = {record["sample_id"] for record in load_manifest()}
    runs = []
    for directory, model_key in (("qwen3", "qwen3"), ("qwen2", "qwen2-historical")):
        predictions = load_jsonl(PUBLISHED_RESULTS / directory / "predictions.jsonl")
        metadata = json.loads(
            (PUBLISHED_RESULTS / directory / "run_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        summary = summarize_records(predictions)

        assert len(predictions) == 280
        assert {record["sample_id"] for record in predictions} == manifest_ids
        assert metadata["status"] == "completed"
        assert metadata["repository_dirty"] is False
        assert metadata["measurement"]["runtime_error_count"] == 0
        assert metadata["runtime"]["model_commit"] == MODEL_SPECS[model_key]["revision"]
        assert (
            metadata["runtime"]["tokenizer_commit"]
            == MODEL_SPECS[model_key]["revision"]
        )
        assert metadata["results"] == summary
        runs.append(predictions)

    published_comparison = json.loads(
        (PUBLISHED_RESULTS / "comparison.json").read_text(encoding="utf-8")
    )
    assert published_comparison["runs"] == [
        summarize_records(records) for records in runs
    ]
    assert published_comparison["paired_comparisons"] == paired_comparisons(runs)


def test_cpu_metadata_records_architecture_and_available_core_count():
    metadata = cpu_metadata()

    assert metadata["architecture"]
    assert metadata["logical_cores"] is None or metadata["logical_cores"] > 0
    assert metadata["model_name"] is None or metadata["model_name"].strip()
