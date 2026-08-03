import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_qwen_sample_manifest import selected_indices
from scripts.qwen_benchmark import (
    MODEL_SPECS,
    paired_bootstrap_interval,
    paired_comparisons,
    summarize_records,
    validate_insertion_only,
    validate_manifest,
    wilson_interval,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "benchmarks/qwen/samples.jsonl"
MANIFEST_SHA256 = "743e7519eb4ef760f45a7b5b6a34fea3b0f7394b85e9fed7609b27864cd8497d"


def load_manifest():
    return [
        json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    ]


def prediction(sample_id, *, model="left", valid=True, correct=False, group="English"):
    return {
        "sample_id": sample_id,
        "model_label": model,
        "model_id": f"example/{model}",
        "model_revision": "a" * 40,
        "valid": valid,
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
        ("CamelCase", "Camel Case", (True, "Camel Case", None)),
        ("CamelCase", "camel Case", (False, None, "changed_non_space_characters")),
        ("icecream", "ice\ncream", (False, None, "changed_non_space_characters")),
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
def test_output_contract_never_repairs_or_falls_back(source, raw_output, expected):
    assert validate_insertion_only(source, raw_output) == expected


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


def test_wilson_interval_and_paired_bootstrap_are_bounded_and_deterministic():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.236593, abs=1e-6)
    assert high == pytest.approx(0.763407, abs=1e-6)
    assert paired_bootstrap_interval([True] * 5, [False] * 5) == (1.0, 1.0)
    assert paired_bootstrap_interval([True, False], [True, False]) == (0.0, 0.0)


def test_summary_counts_invalid_outputs_as_incorrect_and_reports_rate_ci():
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
    assert summary["runtime_error_rate"]["successes"] == 0
    assert summary["throughput_items_per_second"] == pytest.approx(100.0)


def test_summary_separates_runtime_errors_from_invalid_model_outputs():
    record = prediction("one", valid=False, correct=False)
    record["error"] = "RuntimeError: backend failed"
    record["generation_ms"] = None

    summary = summarize_records([record])["overall"]

    assert summary["invalid_output_rate"]["successes"] == 0
    assert summary["runtime_error_rate"]["successes"] == 1


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


def test_model_pins_preserve_qwen2_and_disable_qwen3_thinking():
    assert MODEL_SPECS["qwen3"] == {
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "label": "Qwen3-0.6B (text-only, non-thinking)",
        "enable_thinking": False,
        "status": "current-fallback",
    }
    assert MODEL_SPECS["qwen2-historical"]["model_id"] == "Qwen/Qwen2-0.5B-Instruct"
    assert MODEL_SPECS["qwen2-historical"]["status"] == "historical-reproduction"
