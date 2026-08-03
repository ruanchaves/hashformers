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
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
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
        "label": "Qwen2-0.5B-Instruct (historical reproduction)",
        "enable_thinking": None,
        "status": "historical-reproduction",
    },
}

SYSTEM_PROMPT = (
    "Segment the supplied concatenated string into words. Return exactly the "
    "original characters in their original order, inserting ASCII spaces only "
    "at word boundaries. Do not add, remove, reorder, or change any non-space "
    "character. Return no explanation."
)


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


def validate_insertion_only(
    source: str, raw_output: str | None
) -> tuple[bool, str | None, str | None]:
    """Enforce that an output differs from its source only by inserted spaces.

    Invalid generations are never repaired or silently replaced with the input.
    The returned prediction is whitespace-normalized only after the strict
    contract succeeds; the exact generation remains in ``raw_output``.
    """

    if raw_output is None:
        return False, None, "missing_output"
    if not raw_output:
        return False, None, "empty_output"
    if raw_output.replace(" ", "") != source:
        return False, None, "changed_non_space_characters"
    prediction = " ".join(raw_output.split(" "))
    prediction = " ".join(part for part in prediction.split(" ") if part)
    if not prediction:
        return False, None, "empty_output"
    return True, prediction, None


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
    sample_ids = [record.get("sample_id") for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("prediction file contains duplicate sample IDs")
    model_labels = {record.get("model_label") for record in records}
    if len(model_labels) != 1:
        raise ValueError("prediction file must contain exactly one model label")

    def summarize_subset(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(record.get("correct")) for record in subset)
        invalid = sum(
            not bool(record.get("valid")) and not record.get("error")
            for record in subset
        )
        runtime_errors = sum(bool(record.get("error")) for record in subset)
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
            "invalid_output_rate": rate_summary(invalid, len(subset)),
            "runtime_error_rate": rate_summary(runtime_errors, len(subset)),
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
        "model_label": next(iter(model_labels)),
        "model_id": records[0].get("model_id"),
        "model_revision": records[0].get("model_revision"),
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
        left_by_id = {str(record["sample_id"]): record for record in left}
        for right in runs[left_index + 1 :]:
            right_by_id = {str(record["sample_id"]): record for record in right}
            shared_ids = sorted(left_by_id.keys() & right_by_id.keys())
            if not shared_ids:
                continue
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
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


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


def resolve_torch_dtype(torch: Any, precision: str) -> Any:
    """Map the CLI precision name to a torch dtype."""

    if precision == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[precision]


def cuda_sync(torch: Any) -> None:
    """Synchronize CUDA when it is active so timings include completed kernels."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_messages(source: str) -> list[dict[str, str]]:
    """Build the zero-shot prompt shared by every language and task group."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source},
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
        spec["model_id"], revision=spec["revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"],
        revision=spec["revision"],
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

    runtime = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_map": args.device,
        "requested_precision": args.precision,
        "actual_parameter_dtype": actual_dtype,
        "quantization": args.quantization,
        "model_commit": getattr(model.config, "_commit_hash", None) or spec["revision"],
        "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash")
        or spec["revision"],
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
    inputs = tokenizer(prompt, return_tensors="pt")
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = inputs.to(model_device)
    preprocessing_ms = (time.perf_counter() - preprocessing_start) * 1000

    cuda_sync(torch)
    generation_start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    cuda_sync(torch)
    generation_ms = (time.perf_counter() - generation_start) * 1000
    prompt_tokens = inputs["input_ids"].shape[1]
    generated = output_ids[0][prompt_tokens:]
    raw_output = tokenizer.decode(generated, skip_special_tokens=True)
    return raw_output, preprocessing_ms, generation_ms, len(generated)


def memory_snapshot(torch: Any) -> dict[str, int | None]:
    """Report measured CUDA allocation separately from model-loading memory."""

    if not torch.cuda.is_available():
        return {
            "baseline_allocated_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "baseline_allocated_bytes": torch.cuda.memory_allocated(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
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
    spec = dict(MODEL_SPECS[args.model])
    if args.revision:
        spec["revision"] = args.revision
    predictions_path, metadata_path = prepare_output_directory(
        args.output_dir.resolve(), args.overwrite
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "loading-model",
        "started_at": utc_now(),
        "repository_revision": git_revision(),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "sample_count": len(manifest),
        "model": spec,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "prompt": SYSTEM_PROMPT,
            "few_shot_examples": 0,
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
            "machine": platform.machine(),
            "processor": platform.processor(),
            "gpus_from_nvidia_smi": gpu_driver_metadata(),
        },
    }
    write_json(metadata_path, metadata)
    torch, tokenizer, model, runtime = load_model(args, spec)
    metadata["runtime"] = runtime

    warmup_ids = []
    for sample in manifest[: args.warmup]:
        generate_once(
            torch, tokenizer, model, spec, sample["input"], args.max_new_tokens
        )
        warmup_ids.append(sample["sample_id"])
    cuda_sync(torch)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    baseline_memory = memory_snapshot(torch)
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
                valid, prediction, invalid_reason = validate_insertion_only(
                    sample["input"], raw_output
                )
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
                error = f"{type(exc).__name__}: {exc}"
            correct = bool(
                valid
                and normalize_segmentation(prediction or "")
                == normalize_segmentation(sample["gold"])
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                **sample,
                "model_label": spec["label"],
                "model_id": spec["model_id"],
                "model_revision": runtime["model_commit"],
                "raw_output": raw_output,
                "prediction": prediction,
                "valid": valid,
                "invalid_reason": invalid_reason,
                "correct": correct,
                "preprocessing_ms": preprocessing_ms,
                "generation_ms": generation_ms,
                "generated_tokens": generated_tokens,
                "error": error,
            }
            output_handle.write(canonical_json(record) + "\n")
            output_handle.flush()
    cuda_sync(torch)
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
    memory = memory_snapshot(torch)
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
        "--device", default="auto", help="Transformers device_map value (default: auto)"
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
