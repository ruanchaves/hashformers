#!/usr/bin/env python3
"""Reproducible prompted-Qwen word-segmentation benchmark.

The inference dependencies are imported lazily so the output-contract and
statistics helpers can be tested without installing PyTorch or Transformers.
Run one model per process: this keeps latency and peak-memory measurements from
being contaminated by another resident model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3
PROTOCOL_ID = "hashformers-qwen-space-insertion-v3"
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("benchmarks/qwen/samples.jsonl")
MODEL_SPECS = {
    "qwen3": {
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "label": "Qwen3-0.6B (text-only, non-thinking)",
        "enable_thinking": False,
        "status": "current-fallback",
    },
    "qwen2-historical": {
        "model_id": "Qwen/Qwen2-0.5B-Instruct",
        "revision": "c540970f9e29518b1d8f06ab8b24cba66ad77b6d",
        "label": "Qwen2-0.5B-Instruct (refreshed protocol)",
        "enable_thinking": None,
        "status": "historical-model-under-refreshed-protocol",
    },
}

SYSTEM_PROMPT = (
    "You perform word segmentation. For each user message of the form "
    "'Input: TEXT', return TEXT with ASCII spaces inserted at word boundaries. "
    "Preserve every non-space character, its case, and its order exactly. If "
    "TEXT is one word, return it unchanged. Return plain text only, without "
    "quotes, labels, backticks, code fences, or explanation."
)
USER_PROMPT_TEMPLATE = "Input: {source}"


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for hashing and line-oriented output."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSON Lines file and report its physical line on errors."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def validate_manifest(records: Sequence[Mapping[str, Any]]) -> None:
    """Validate fields required to make predictions auditable and pairable."""

    required = {
        "sample_id",
        "dataset",
        "dataset_revision",
        "split",
        "row_index",
        "group",
        "input",
        "gold",
    }
    seen: set[str] = set()
    for position, record in enumerate(records, 1):
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError(
                f"manifest record {position} is missing: {', '.join(missing)}"
            )
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"manifest record {position} has an invalid sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        for field in ("input", "gold"):
            if not isinstance(record[field], str) or not record[field]:
                raise ValueError(f"{sample_id}: {field} must be a non-empty string")
        for field in ("dataset", "split", "group"):
            if not isinstance(record[field], str) or not record[field]:
                raise ValueError(f"{sample_id}: {field} must be a non-empty string")
        validate_revision(record["dataset_revision"], "dataset revision")
        row_index = record["row_index"]
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or row_index < 0
        ):
            raise ValueError(f"{sample_id}: row_index must be a non-negative integer")
        if record["gold"].replace(" ", "") != record["input"]:
            raise ValueError(
                f"{sample_id}: gold must differ from input only by inserted spaces"
            )


def validate_revision(revision: str, name: str = "model revision") -> str:
    """Require an immutable lowercase Hub commit SHA."""

    if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{name} must be an exact 40-character Hub commit SHA")
    return revision


def resolve_hub_file_commit(
    model_id: str,
    revision: str,
    filename: str,
    *,
    download_file: Any | None = None,
) -> str:
    """Resolve the immutable commit backing one downloaded Hub file.

    Transformers 5 no longer preserves ``_commit_hash`` in tokenizer
    ``init_kwargs``.  Hugging Face Hub still returns the downloaded artifact
    through its ``snapshots/<commit>/`` cache path, so use that path as the
    version-independent source of tokenizer provenance.
    """

    if download_file is None:
        from huggingface_hub import hf_hub_download

        download_file = hf_hub_download
    cached_path = Path(
        download_file(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            token=False,
        )
    )
    parts = cached_path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and index + 1 < len(parts):
            commit = parts[index + 1]
            if REVISION_PATTERN.fullmatch(commit) is not None:
                return commit
    raise RuntimeError(
        f"could not resolve an immutable Hub commit from cached {filename}: "
        f"{cached_path}"
    )


def single_device(value: str) -> str:
    """Parse one explicit CPU or CUDA device for isolated measurements."""

    normalized = value.strip().lower()
    if normalized == "cpu":
        return normalized
    if normalized == "cuda":
        return "cuda:0"
    if re.fullmatch(r"cuda:[0-9]+", normalized):
        return normalized
    raise argparse.ArgumentTypeError("device must be cpu, cuda, or cuda:<index>")


def parse_insertion_only(
    source: str, raw_output: str | None
) -> tuple[bool, str | None, str | None, str | None]:
    """Parse a minimal response envelope, then enforce insertion-only content.

    A model may wrap its entire answer in one matching pair of ASCII quotes.
    That presentation envelope is recorded separately and removed only when
    the enclosed text already satisfies the insertion-only contract. Invalid
    generations are not repaired by this strict parser. Recovery and fallback
    are handled separately by :func:`propose_segmentation`, while the exact
    generation remains in ``raw_output``.
    """

    if raw_output is None:
        return False, None, "missing_output", None
    if not raw_output:
        return False, None, "empty_output", None

    candidates: list[tuple[str, str | None]] = [(raw_output, None)]
    stripped = raw_output.strip(" ")
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        candidates.append((stripped[1:-1], "matching_ascii_quotes"))

    for candidate, output_wrapper in candidates:
        if candidate.replace(" ", "") != source:
            continue
        prediction = " ".join(part for part in candidate.split(" ") if part)
        if prediction:
            return True, prediction, None, output_wrapper
    return False, None, "changed_non_space_characters", None


RECOVERY_LABEL_PATTERN = re.compile(
    r"^(?:output|result|answer|segmentation|segmented(?:\s+text)?):\s*(.*)$",
    re.IGNORECASE,
)


def _recovery_candidates(raw_output: str) -> list[str]:
    """Extract bounded candidate answers from a non-conforming generation."""

    stripped = raw_output.strip()
    if not stripped:
        return []

    candidates: list[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1].strip()
        elif value.startswith('\\"') and value.endswith('\\"') and len(value) >= 4:
            value = value[2:-2].strip()
        if value and value not in candidates:
            candidates.append(value)

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if stripped.startswith("```") and stripped.endswith("```") and len(lines) >= 2:
        lines = lines[1:-1]
        if lines and lines[0].casefold() in {"text", "plaintext", "txt"}:
            lines = lines[1:]

    if len(lines) == 1:
        match = RECOVERY_LABEL_PATTERN.fullmatch(lines[0])
        add(match.group(1) if match else lines[0])
    else:
        for index, line in enumerate(lines):
            match = RECOVERY_LABEL_PATTERN.fullmatch(line)
            if match and match.group(1):
                add(match.group(1))
            elif match and index + 1 < len(lines):
                add(lines[index + 1])
            elif ":" not in line:
                add(line)
    return candidates


def _candidate_boundaries(candidate: str) -> tuple[str, list[int]]:
    """Return non-separator characters and proposed boundaries between them."""

    characters: list[str] = []
    boundaries: list[int] = []
    pending_boundary = False
    for character in candidate:
        if character.isspace() or character == "_":
            pending_boundary = bool(characters)
            continue
        if pending_boundary and characters:
            boundaries.append(len(characters))
        characters.append(character)
        pending_boundary = False
    compact = "".join(characters)
    return compact, sorted(set(boundaries))


def _project_candidate_boundaries(
    source: str, candidate: str
) -> tuple[str, str, float] | None:
    """Align candidate characters to source and project its space boundaries.

    Character edits are never copied into the proposal. A global edit-distance
    alignment maps only the candidate's boundary locations back onto the exact
    original source string. Candidates with no boundary signal, no characters,
    or more than 50% normalized edit distance are rejected.
    """

    compact, candidate_boundaries = _candidate_boundaries(candidate)
    if not compact or not candidate_boundaries:
        return None

    source_length = len(source)
    candidate_length = len(compact)
    distance: list[list[int]] = [
        [0] * (candidate_length + 1) for _ in range(source_length + 1)
    ]
    operation: list[list[str | None]] = [
        [None] * (candidate_length + 1) for _ in range(source_length + 1)
    ]
    for source_index in range(1, source_length + 1):
        distance[source_index][0] = source_index
        operation[source_index][0] = "delete_source"
    for candidate_index in range(1, candidate_length + 1):
        distance[0][candidate_index] = candidate_index
        operation[0][candidate_index] = "insert_candidate"

    for source_index in range(1, source_length + 1):
        for candidate_index in range(1, candidate_length + 1):
            substitution_cost = int(
                source[source_index - 1].casefold()
                != compact[candidate_index - 1].casefold()
            )
            choices = (
                (
                    distance[source_index - 1][candidate_index - 1] + substitution_cost,
                    0,
                    "diagonal",
                ),
                (distance[source_index - 1][candidate_index] + 1, 1, "delete_source"),
                (
                    distance[source_index][candidate_index - 1] + 1,
                    2,
                    "insert_candidate",
                ),
            )
            best_cost, _, best_operation = min(choices)
            distance[source_index][candidate_index] = best_cost
            operation[source_index][candidate_index] = best_operation

    edit_distance = distance[source_length][candidate_length]
    normalized_distance = edit_distance / max(source_length, candidate_length)
    if normalized_distance > 0.5:
        return None

    reversed_operations: list[str] = []
    source_index = source_length
    candidate_index = candidate_length
    while source_index or candidate_index:
        current = operation[source_index][candidate_index]
        if current is None:
            raise RuntimeError("alignment traceback reached an incomplete cell")
        reversed_operations.append(current)
        if current == "diagonal":
            source_index -= 1
            candidate_index -= 1
        elif current == "delete_source":
            source_index -= 1
        else:
            candidate_index -= 1

    candidate_prefix_to_source = {0: 0}
    source_index = 0
    candidate_index = 0
    for current in reversed(reversed_operations):
        if current in {"diagonal", "delete_source"}:
            source_index += 1
        if current in {"diagonal", "insert_candidate"}:
            candidate_index += 1
        candidate_prefix_to_source[candidate_index] = source_index

    projected_boundaries = sorted(
        {
            candidate_prefix_to_source[boundary]
            for boundary in candidate_boundaries
            if 0 < candidate_prefix_to_source[boundary] < source_length
        }
    )
    if not projected_boundaries:
        return None
    pieces = []
    previous = 0
    for boundary in projected_boundaries:
        pieces.append(source[previous:boundary])
        previous = boundary
    pieces.append(source[previous:])
    proposal = " ".join(piece for piece in pieces if piece)
    method = (
        "case_preserving_projection"
        if candidate_length == source_length
        and all(
            left.casefold() == right.casefold() for left, right in zip(source, compact)
        )
        else "edit_alignment_projection"
    )
    return proposal, method, normalized_distance


def propose_segmentation(
    source: str, raw_output: str | None
) -> tuple[
    bool,
    str,
    str | None,
    str | None,
    str,
    str | None,
]:
    """Return strict validity plus an always-available segmentation proposal.

    Strictly valid model output is used directly. Otherwise, the closest
    bounded answer candidate with a usable boundary signal is aligned onto the
    original source characters. If no such signal exists, the unchanged source
    is returned as an explicit, measurable fallback.
    """

    valid, prediction, invalid_reason, output_wrapper = parse_insertion_only(
        source, raw_output
    )
    if valid:
        assert prediction is not None
        return (
            True,
            prediction,
            None,
            output_wrapper,
            "model_output",
            None,
        )

    projected = []
    for position, candidate in enumerate(_recovery_candidates(raw_output or "")):
        result = _project_candidate_boundaries(source, candidate)
        if result is not None:
            proposal, method, normalized_distance = result
            projected.append((normalized_distance, position, proposal, method))
    if projected:
        _, _, proposal, method = min(projected)
        return (
            False,
            proposal,
            invalid_reason,
            None,
            "recovered_model_output",
            method,
        )
    return (
        False,
        source,
        invalid_reason,
        None,
        "source_fallback",
        "unchanged_input",
    )


def validate_insertion_only(
    source: str, raw_output: str | None
) -> tuple[bool, str | None, str | None]:
    """Return the insertion-only validation fields used by public callers."""

    valid, prediction, invalid_reason, _ = parse_insertion_only(source, raw_output)
    return valid, prediction, invalid_reason


def normalize_segmentation(value: str) -> str:
    """Normalize spacing and case for the report's exact-match metric."""

    return " ".join(value.split()).casefold()


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Calculate a two-sided Wilson score interval for a binomial rate."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must describe a binomial count")
    if total == 0:
        return (math.nan, math.nan)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile from sorted values."""

    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = probability * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_interval(
    left: Sequence[bool],
    right: Sequence[bool],
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a paired percentile-bootstrap CI for accuracy(left)-accuracy(right)."""

    if len(left) != len(right):
        raise ValueError("paired samples must have the same length")
    if not left:
        return (math.nan, math.nan)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    differences = [int(a) - int(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    size = len(differences)
    estimates = []
    for _ in range(iterations):
        estimates.append(
            sum(differences[rng.randrange(size)] for _ in range(size)) / size
        )
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def rate_summary(successes: int, total: int) -> dict[str, Any]:
    """Represent a rate and its 95% Wilson interval."""

    low, high = wilson_interval(successes, total)
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else None,
        "ci_95": [low, high] if total else [None, None],
        "ci_method": "Wilson score",
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize accuracy and invalid-output rates for one prediction run."""

    if not records:
        raise ValueError("prediction file is empty")
    required_fields = {
        "schema_version",
        "protocol_id",
        "manifest_sha256",
        "sample_id",
        "dataset",
        "dataset_revision",
        "split",
        "row_index",
        "group",
        "input",
        "gold",
        "model_label",
        "model_id",
        "model_revision",
        "requested_precision",
        "actual_parameter_dtype",
        "quantization",
        "resolved_device",
        "raw_output",
        "output_wrapper",
        "prediction",
        "prediction_source",
        "recovery_method",
        "valid",
        "invalid_reason",
        "strict_correct",
        "correct",
        "generation_ms",
        "error",
    }
    for position, record in enumerate(records, 1):
        missing = sorted(required_fields.difference(record))
        if missing:
            raise ValueError(
                f"prediction record {position} is missing: {', '.join(missing)}"
            )
    validate_manifest(records)

    singleton_fields = (
        "protocol_id",
        "manifest_sha256",
        "model_label",
        "model_id",
        "model_revision",
        "requested_precision",
        "actual_parameter_dtype",
        "quantization",
        "resolved_device",
    )
    for position, record in enumerate(records, 1):
        sample_id = record["sample_id"]
        if record["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"{sample_id}: unsupported prediction schema version")
        for field in singleton_fields:
            if not isinstance(record[field], str) or not record[field]:
                raise ValueError(f"prediction record {position} has an invalid {field}")
        if (
            not isinstance(record["valid"], bool)
            or not isinstance(record["strict_correct"], bool)
            or not isinstance(record["correct"], bool)
        ):
            raise TypeError(
                f"{sample_id}: valid, strict_correct, and correct must be booleans"
            )
        for field in (
            "raw_output",
            "output_wrapper",
            "prediction",
            "prediction_source",
            "recovery_method",
            "invalid_reason",
            "error",
        ):
            if record[field] is not None and not isinstance(record[field], str):
                raise ValueError(f"{sample_id}: {field} must be text or null")
        if record["output_wrapper"] not in (None, "matching_ascii_quotes"):
            raise ValueError(f"{sample_id}: unsupported output_wrapper")
        if record["prediction_source"] not in (
            None,
            "model_output",
            "recovered_model_output",
            "source_fallback",
        ):
            raise ValueError(f"{sample_id}: unsupported prediction_source")
        if record["recovery_method"] not in (
            None,
            "case_preserving_projection",
            "edit_alignment_projection",
            "unchanged_input",
        ):
            raise ValueError(f"{sample_id}: unsupported recovery_method")
        generation_ms = record["generation_ms"]
        if generation_ms is not None and (
            isinstance(generation_ms, bool)
            or not isinstance(generation_ms, (int, float))
            or not math.isfinite(float(generation_ms))
            or generation_ms < 0
        ):
            raise ValueError(
                f"{sample_id}: generation_ms must be a non-negative number or null"
            )
        if record["error"]:
            if (
                any(
                    record[field] is not None
                    for field in (
                        "raw_output",
                        "output_wrapper",
                        "prediction",
                        "prediction_source",
                        "recovery_method",
                    )
                )
                or record["invalid_reason"] != "runtime_error"
            ):
                raise ValueError(f"{sample_id}: inconsistent runtime-error record")
        else:
            (
                valid,
                prediction,
                invalid_reason,
                output_wrapper,
                prediction_source,
                recovery_method,
            ) = propose_segmentation(record["input"], record["raw_output"])
            expected_fields = {
                "valid": valid,
                "prediction": prediction,
                "invalid_reason": invalid_reason,
                "output_wrapper": output_wrapper,
                "prediction_source": prediction_source,
                "recovery_method": recovery_method,
            }
            if any(record[field] != value for field, value in expected_fields.items()):
                raise ValueError(f"{sample_id}: inconsistent segmentation proposal")
        expected_correct = bool(
            record["prediction"] is not None
            and normalize_segmentation(record["prediction"])
            == normalize_segmentation(record["gold"])
        )
        expected_strict_correct = bool(record["valid"] and expected_correct)
        if record["strict_correct"] != expected_strict_correct:
            raise ValueError(
                f"{sample_id}: strict_correct does not match strict validation"
            )
        if record["correct"] != expected_correct:
            raise ValueError(f"{sample_id}: correct does not match the prediction")

    identities = {
        field: {record[field] for record in records} for field in singleton_fields
    }
    for field, values in identities.items():
        if len(values) != 1:
            raise ValueError(f"prediction file must contain exactly one {field}")
    validate_revision(records[0]["model_revision"])
    if re.fullmatch(r"[0-9a-f]{64}", records[0]["manifest_sha256"]) is None:
        raise ValueError("manifest_sha256 must be an exact lowercase SHA-256 digest")

    def summarize_subset(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(record.get("correct")) for record in subset)
        strict_correct = sum(bool(record.get("strict_correct")) for record in subset)
        invalid = sum(
            not bool(record.get("valid")) and not record.get("error")
            for record in subset
        )
        runtime_errors = sum(bool(record.get("error")) for record in subset)
        wrapped_outputs = sum(bool(record.get("output_wrapper")) for record in subset)
        recovered_predictions = sum(
            record.get("prediction_source") == "recovered_model_output"
            for record in subset
        )
        source_fallbacks = sum(
            record.get("prediction_source") == "source_fallback" for record in subset
        )
        generation_seconds = (
            sum(float(record.get("generation_ms") or 0.0) for record in subset) / 1000
        )
        latencies = [
            float(record["generation_ms"])
            for record in subset
            if record.get("generation_ms") is not None
        ]
        return {
            "count": len(subset),
            "accuracy": rate_summary(correct, len(subset)),
            "strict_output_accuracy": rate_summary(strict_correct, len(subset)),
            "invalid_output_rate": rate_summary(invalid, len(subset)),
            "runtime_error_rate": rate_summary(runtime_errors, len(subset)),
            "output_wrapper_rate": rate_summary(wrapped_outputs, len(subset)),
            "recovered_prediction_rate": rate_summary(
                recovered_predictions, len(subset)
            ),
            "source_fallback_rate": rate_summary(source_fallbacks, len(subset)),
            "latency_ms": {
                "mean": statistics.fmean(latencies) if latencies else None,
                "median": statistics.median(latencies) if latencies else None,
                "p95": percentile(sorted(latencies), 0.95) if latencies else None,
            },
            "throughput_items_per_second": len(latencies) / generation_seconds
            if generation_seconds
            else None,
        }

    groups = sorted({str(record.get("group")) for record in records})
    return {
        "protocol_id": records[0]["protocol_id"],
        "manifest_sha256": records[0]["manifest_sha256"],
        "model_label": records[0]["model_label"],
        "model_id": records[0]["model_id"],
        "model_revision": records[0]["model_revision"],
        "requested_precision": records[0]["requested_precision"],
        "actual_parameter_dtype": records[0]["actual_parameter_dtype"],
        "quantization": records[0]["quantization"],
        "resolved_device": records[0]["resolved_device"],
        "overall": summarize_subset(records),
        "groups": {
            group: summarize_subset(
                [record for record in records if str(record.get("group")) == group]
            )
            for group in groups
        },
    }


def paired_comparisons(
    runs: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Calculate deterministic paired accuracy-difference confidence intervals."""

    comparisons: list[dict[str, Any]] = []
    for left_index, left in enumerate(runs):
        summarize_records(left)
        left_by_id = {str(record["sample_id"]): record for record in left}
        for right in runs[left_index + 1 :]:
            summarize_records(right)
            right_by_id = {str(record["sample_id"]): record for record in right}
            if set(left_by_id) != set(right_by_id):
                raise ValueError("paired runs must contain exactly the same sample IDs")
            for field in ("protocol_id", "manifest_sha256"):
                if left[0][field] != right[0][field]:
                    raise ValueError(f"paired runs must use the same {field}")
            shared_ids = sorted(left_by_id)
            provenance_fields = (
                "dataset",
                "dataset_revision",
                "split",
                "row_index",
                "group",
                "input",
                "gold",
            )
            for sample_id in shared_ids:
                for field in provenance_fields:
                    if left_by_id[sample_id].get(field) != right_by_id[sample_id].get(
                        field
                    ):
                        raise ValueError(
                            f"paired sample {sample_id!r} differs in {field}"
                        )
            left_values = [
                bool(left_by_id[sample_id].get("correct")) for sample_id in shared_ids
            ]
            right_values = [
                bool(right_by_id[sample_id].get("correct")) for sample_id in shared_ids
            ]
            difference = statistics.fmean(
                int(a) - int(b) for a, b in zip(left_values, right_values)
            )
            low, high = paired_bootstrap_interval(left_values, right_values)
            comparisons.append(
                {
                    "left": left[0].get("model_label"),
                    "right": right[0].get("model_label"),
                    "protocol_id": left[0]["protocol_id"],
                    "manifest_sha256": left[0]["manifest_sha256"],
                    "left_configuration": {
                        field: left[0][field]
                        for field in (
                            "model_id",
                            "model_revision",
                            "requested_precision",
                            "actual_parameter_dtype",
                            "quantization",
                            "resolved_device",
                        )
                    },
                    "right_configuration": {
                        field: right[0][field]
                        for field in (
                            "model_id",
                            "model_revision",
                            "requested_precision",
                            "actual_parameter_dtype",
                            "quantization",
                            "resolved_device",
                        )
                    },
                    "paired_samples": len(shared_ids),
                    "accuracy_difference": difference,
                    "ci_95": [low, high],
                    "ci_method": "paired percentile bootstrap (10,000 resamples, seed 42)",
                }
            )
    return comparisons


def write_json(path: Path, value: Any) -> None:
    """Atomically write pretty JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def git_revision() -> str | None:
    """Return the current repository revision when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=REPOSITORY_ROOT,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def git_dirty() -> bool | None:
    """Report tracked working-tree changes for the benchmark source revision."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            cwd=REPOSITORY_ROOT,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def package_version(distribution: str) -> str | None:
    """Read one installed distribution version without importing it."""

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def gpu_driver_metadata() -> list[dict[str, str]]:
    """Read GPU model, memory, and driver metadata without failing a CPU run."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            devices.append(
                {"name": fields[0], "memory_mib": fields[1], "driver": fields[2]}
            )
    return devices


def cpu_metadata() -> dict[str, Any]:
    """Return a stable CPU description even when ``platform.processor`` is empty."""

    model_name = platform.processor().strip() or None
    cpuinfo = Path("/proc/cpuinfo")
    if model_name is None and cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                field, separator, value = line.partition(":")
                if separator and field.strip() in {"model name", "Hardware"}:
                    model_name = value.strip() or None
                    if model_name is not None:
                        break
        except OSError:
            pass
    return {
        "model_name": model_name,
        "architecture": platform.machine(),
        "logical_cores": os.cpu_count(),
    }


def resolve_torch_dtype(torch: Any, precision: str) -> Any:
    """Map the CLI precision name to a torch dtype."""

    if precision == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[precision]


def cuda_sync(torch: Any, device: str) -> None:
    """Synchronize CUDA when it is active so timings include completed kernels."""

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def build_messages(source: str) -> list[dict[str, str]]:
    """Build the zero-shot prompt shared by every language and task group."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(source=source)},
    ]


def load_model(
    args: argparse.Namespace, spec: Mapping[str, Any]
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load one pinned causal LM and return its exact runtime configuration."""

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "run requires PyTorch, Transformers >=4.51, and Accelerate; "
            "see benchmarks/qwen/README.md"
        ) from exc

    dtype = resolve_torch_dtype(torch, args.precision)
    quantization_config = None
    if args.quantization == "bnb-4bit-nf4":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit("bnb-4bit-nf4 requires bitsandbytes") from exc
        compute_dtype = dtype if dtype != "auto" else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        spec["model_id"],
        revision=spec["revision"],
        token=False,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"],
        revision=spec["revision"],
        token=False,
        trust_remote_code=False,
        device_map=args.device,
        torch_dtype=dtype,
        quantization_config=quantization_config,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        actual_dtype = str(next(model.parameters()).dtype)
    except StopIteration:
        actual_dtype = None

    model_commit = getattr(model.config, "_commit_hash", None)
    tokenizer_commit = tokenizer.init_kwargs.get("_commit_hash")
    if tokenizer_commit is None:
        tokenizer_commit = resolve_hub_file_commit(
            spec["model_id"], spec["revision"], "tokenizer_config.json"
        )
    model_device = str(getattr(model, "device", ""))
    if model_device == "cuda":
        model_device = "cuda:0"
    if model_device != args.device:
        raise RuntimeError(
            f"model resolved to {model_device!r}, expected isolated device "
            f"{args.device!r}"
        )
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        mapped_devices = {
            f"cuda:{value}" if isinstance(value, int) else str(value)
            for value in device_map.values()
        }
        mapped_devices.discard("cuda")
        if mapped_devices and mapped_devices != {args.device}:
            raise RuntimeError(
                "model must be resident on exactly one requested device; "
                f"resolved map uses {sorted(mapped_devices)}"
            )

    runtime = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "accelerate_version": package_version("accelerate"),
        "bitsandbytes_version": package_version("bitsandbytes"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": args.device,
        "resolved_device": model_device,
        "resolved_device_map": (
            {key: str(value) for key, value in device_map.items()}
            if isinstance(device_map, Mapping)
            else None
        ),
        "requested_precision": args.precision,
        "actual_parameter_dtype": actual_dtype,
        "quantization": args.quantization,
        "model_commit": model_commit,
        "tokenizer_commit": tokenizer_commit,
    }
    if model_device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(model_device)
        runtime["cuda_device"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return torch, tokenizer, model, runtime


def generate_once(
    torch: Any,
    tokenizer: Any,
    model: Any,
    spec: Mapping[str, Any],
    source: str,
    max_new_tokens: int,
) -> tuple[str, float, float, int]:
    """Generate one raw response and return preprocessing/inference timings."""

    template_options: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if spec["enable_thinking"] is not None:
        template_options["enable_thinking"] = spec["enable_thinking"]
    preprocessing_start = time.perf_counter()
    prompt = tokenizer.apply_chat_template(build_messages(source), **template_options)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = inputs.to(model_device)
    preprocessing_ms = (time.perf_counter() - preprocessing_start) * 1000

    resolved_device = str(model_device)
    if resolved_device == "cuda":
        resolved_device = "cuda:0"
    cuda_sync(torch, resolved_device)
    generation_start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    cuda_sync(torch, resolved_device)
    generation_ms = (time.perf_counter() - generation_start) * 1000
    prompt_tokens = inputs["input_ids"].shape[1]
    generated = output_ids[0][prompt_tokens:]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True)
    return raw_output, preprocessing_ms, generation_ms, len(generated)


def memory_snapshot(torch: Any, device: str) -> dict[str, int | None]:
    """Report measured CUDA allocation separately from model-loading memory."""

    if not device.startswith("cuda"):
        return {
            "baseline_allocated_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "baseline_allocated_bytes": torch.cuda.memory_allocated(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def prepare_output_directory(path: Path, overwrite: bool) -> tuple[Path, Path]:
    """Create a run directory while protecting prior benchmark artifacts."""

    path.mkdir(parents=True, exist_ok=True)
    predictions = path / "predictions.jsonl"
    metadata = path / "run_metadata.json"
    existing = [
        candidate for candidate in (predictions, metadata) if candidate.exists()
    ]
    if existing and not overwrite:
        names = ", ".join(str(candidate) for candidate in existing)
        raise SystemExit(f"refusing to replace existing benchmark artifacts: {names}")
    for candidate in existing:
        candidate.unlink()
    return predictions, metadata


def run_benchmark(args: argparse.Namespace) -> None:
    """Execute one isolated, pinned model run over the committed manifest."""

    manifest_path = args.manifest.resolve()
    manifest = load_jsonl(manifest_path)
    validate_manifest(manifest)
    manifest_sha256 = file_sha256(manifest_path)
    spec = dict(MODEL_SPECS[args.model])
    if args.revision:
        spec["revision"] = validate_revision(args.revision)
    else:
        validate_revision(spec["revision"])
    predictions_path, metadata_path = prepare_output_directory(
        args.output_dir.resolve(), args.overwrite
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "loading-model",
        "started_at": utc_now(),
        "repository_revision": git_revision(),
        "repository_dirty": git_dirty(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "sample_count": len(manifest),
        "model": spec,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": USER_PROMPT_TEMPLATE,
            "few_shot_examples": 0,
            "accepted_output_envelopes": [
                "plain_text",
                "one_matching_pair_of_ascii_quotes",
            ],
            "invalid_output_handling": {
                "strict_validity_reported_separately": True,
                "recovery": (
                    "extract a bounded answer candidate and project its word "
                    "boundaries onto the original source with global edit alignment"
                ),
                "maximum_normalized_recovery_edit_distance": 0.5,
                "fallback": "unchanged input when no recoverable boundary exists",
                "character_policy": (
                    "proposals always preserve source non-space characters exactly"
                ),
            },
        },
        "measurement": {
            "warmup_items": min(args.warmup, len(manifest)),
            "batch_size": 1,
            "cuda_synchronize_before_and_after_generation_when_cuda_is_active": True,
            "one_model_per_process": True,
            "latency_scope": "model.generate only; preprocessing reported separately",
        },
        "hardware": {
            "platform": platform.platform(),
            "cpu": cpu_metadata(),
            "gpus_from_nvidia_smi": gpu_driver_metadata(),
        },
    }
    write_json(metadata_path, metadata)
    torch, tokenizer, model, runtime = load_model(args, spec)
    for component in ("model_commit", "tokenizer_commit"):
        if runtime[component] != spec["revision"]:
            raise RuntimeError(
                f"{component} does not match requested revision {spec['revision']}"
            )
    metadata["runtime"] = runtime
    runtime_device = runtime["resolved_device"]

    warmup_ids = []
    for sample in manifest[: args.warmup]:
        generate_once(
            torch, tokenizer, model, spec, sample["input"], args.max_new_tokens
        )
        warmup_ids.append(sample["sample_id"])
    cuda_sync(torch, runtime_device)
    if runtime_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(runtime_device)
    baseline_memory = memory_snapshot(torch, runtime_device)
    metadata["measurement"]["warmup_sample_ids"] = warmup_ids
    metadata["measurement"]["baseline_gpu_memory"] = baseline_memory
    metadata["status"] = "running"
    write_json(metadata_path, metadata)

    measured_start = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8", newline="\n") as output_handle:
        for sample in manifest:
            error = None
            try:
                raw_output, preprocessing_ms, generation_ms, generated_tokens = (
                    generate_once(
                        torch,
                        tokenizer,
                        model,
                        spec,
                        sample["input"],
                        args.max_new_tokens,
                    )
                )
                (
                    valid,
                    prediction,
                    invalid_reason,
                    output_wrapper,
                    prediction_source,
                    recovery_method,
                ) = propose_segmentation(sample["input"], raw_output)
            # Record arbitrary backend/model failures per sample so a partial
            # hardware run remains auditable instead of losing prior outputs.
            except Exception as exc:  # noqa: BLE001
                raw_output = None
                preprocessing_ms = None
                generation_ms = None
                generated_tokens = None
                valid = False
                prediction = None
                invalid_reason = "runtime_error"
                output_wrapper = None
                prediction_source = None
                recovery_method = None
                error = f"{type(exc).__name__}: {exc}"
            correct = bool(
                prediction is not None
                and normalize_segmentation(prediction)
                == normalize_segmentation(sample["gold"])
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "manifest_sha256": manifest_sha256,
                **sample,
                "model_label": spec["label"],
                "model_id": spec["model_id"],
                "model_revision": runtime["model_commit"],
                "requested_precision": runtime["requested_precision"],
                "actual_parameter_dtype": runtime["actual_parameter_dtype"],
                "quantization": runtime["quantization"],
                "resolved_device": runtime["resolved_device"],
                "raw_output": raw_output,
                "output_wrapper": output_wrapper,
                "prediction": prediction,
                "prediction_source": prediction_source,
                "recovery_method": recovery_method,
                "valid": valid,
                "invalid_reason": invalid_reason,
                "strict_correct": bool(valid and correct),
                "correct": correct,
                "preprocessing_ms": preprocessing_ms,
                "generation_ms": generation_ms,
                "generated_tokens": generated_tokens,
                "error": error,
            }
            output_handle.write(canonical_json(record) + "\n")
            output_handle.flush()
    cuda_sync(torch, runtime_device)
    measured_seconds = time.perf_counter() - measured_start
    completed_records = load_jsonl(predictions_path)
    runtime_error_count = sum(bool(record.get("error")) for record in completed_records)
    metadata["status"] = (
        "completed" if runtime_error_count == 0 else "completed-with-errors"
    )
    metadata["completed_at"] = utc_now()
    metadata["measurement"]["measured_wall_seconds"] = measured_seconds
    metadata["measurement"]["throughput_items_per_wall_second"] = (
        len(completed_records) / measured_seconds if measured_seconds else None
    )
    memory = memory_snapshot(torch, runtime_device)
    memory["baseline_allocated_bytes"] = baseline_memory["baseline_allocated_bytes"]
    metadata["measurement"]["gpu_memory"] = memory
    metadata["measurement"]["runtime_error_count"] = runtime_error_count
    metadata["results"] = summarize_records(completed_records)
    write_json(metadata_path, metadata)
    print(f"saved {len(completed_records)} raw predictions to {predictions_path}")
    print(f"saved run metadata and metrics to {metadata_path}")


def summarize_files(args: argparse.Namespace) -> None:
    """Summarize one or more raw prediction artifacts with paired intervals."""

    runs = []
    sources = []
    for path in args.predictions:
        resolved = path.resolve()
        records = load_jsonl(resolved)
        if not records:
            raise SystemExit(f"prediction file is empty: {resolved}")
        runs.append(records)
        sources.append({"path": str(resolved), "sha256": file_sha256(resolved)})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "sources": sources,
        "runs": [summarize_records(records) for records in runs],
        "paired_comparisons": paired_comparisons(runs),
    }
    write_json(args.output.resolve(), summary)
    print(f"saved summary to {args.output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one model in an isolated process")
    run.add_argument("--model", choices=sorted(MODEL_SPECS), default="qwen3")
    run.add_argument("--revision", help="override the pinned model revision")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--precision",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    run.add_argument("--quantization", choices=("none", "bnb-4bit-nf4"), default="none")
    run.add_argument(
        "--device",
        type=single_device,
        default="cuda:0",
        help="single isolated device: cpu, cuda, or cuda:<index> (default: cuda:0)",
    )
    run.add_argument("--max-new-tokens", type=int, default=64)
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(handler=run_benchmark)

    summarize = subparsers.add_parser(
        "summarize", help="summarize saved raw predictions"
    )
    summarize.add_argument("--predictions", type=Path, nargs="+", required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.set_defaults(handler=summarize_files)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    if getattr(args, "warmup", 0) < 0:
        raise SystemExit("--warmup cannot be negative")
    if getattr(args, "max_new_tokens", 1) < 1:
        raise SystemExit("--max-new-tokens must be positive")
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
