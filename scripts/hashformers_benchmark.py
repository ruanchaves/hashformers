#!/usr/bin/env python3
"""Run pinned Hashformers beam-search baselines on the fixed issue #78 samples.

This complements the prompted-Qwen runner without pretending that the archived
January results used the same samples. Each model runs in an isolated process;
the ``compare`` command joins saved predictions by stable sample ID and applies
the same case-insensitive exact-segmentation metric used by the Qwen report.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.qwen_benchmark import (
    SCHEMA_VERSION,
    canonical_json,
    cpu_metadata,
    file_sha256,
    git_dirty,
    git_revision,
    gpu_driver_metadata,
    load_jsonl,
    memory_snapshot,
    normalize_segmentation,
    package_version,
    paired_bootstrap_interval,
    prepare_output_directory,
    propose_segmentation,
    single_device,
    summarize_records,
    utc_now,
    validate_manifest,
    validate_revision,
    write_json,
)

PROTOCOL_ID = "hashformers-beam-search-fixed-manifest-v1"
EVALUATION_CONTRACT_ID = "space-insertion-exact-match-v1"
DEFAULT_MANIFEST = Path("benchmarks/qwen/samples.jsonl")
MODEL_SPECS = {
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


def candidate_batch_size(value: str) -> int | str:
    """Parse one positive fixed candidate microbatch size or ``auto``."""

    normalized = value.strip().lower()
    if normalized == "auto":
        return normalized
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "candidate batch size must be a positive integer or auto"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            "candidate batch size must be a positive integer or auto"
        )
    return parsed


def resolve_model_snapshot(
    model_id: str,
    revision: str,
    *,
    download_snapshot: Any | None = None,
) -> Path:
    """Download and verify one immutable Hub model snapshot."""

    validate_revision(revision)
    if download_snapshot is None:
        from huggingface_hub import snapshot_download

        download_snapshot = snapshot_download
    snapshot = Path(
        download_snapshot(
            repo_id=model_id,
            revision=revision,
            token=False,
        )
    ).resolve()
    parts = snapshot.parts
    if "snapshots" not in parts or snapshot.name != revision:
        raise RuntimeError(
            f"model snapshot did not resolve to snapshots/{revision}: {snapshot}"
        )
    return snapshot


def select_manifest(
    records: Sequence[Mapping[str, Any]], scope_dataset: str | None
) -> list[dict[str, Any]]:
    """Apply a model's documented dataset scope without changing sample IDs."""

    selected = [
        dict(record)
        for record in records
        if scope_dataset is None or record["dataset"] == scope_dataset
    ]
    if not selected:
        raise ValueError(f"manifest has no records for dataset {scope_dataset!r}")
    return selected


def load_segmenter(
    args: argparse.Namespace, spec: Mapping[str, Any]
) -> tuple[Any, Any, dict[str, Any]]:
    """Load one pinned Hashformers model on one explicit device."""

    try:
        import torch
        import transformers

        from hashformers import TransformerWordSegmenter
    except ImportError as exc:
        raise SystemExit(
            "run requires Hashformers, minicons, PyTorch, Transformers, and "
            "huggingface_hub installed from the checked-out repository"
        ) from exc

    snapshot = resolve_model_snapshot(spec["model_id"], spec["revision"])
    segmenter = TransformerWordSegmenter(
        segmenter_model_name_or_path=str(snapshot),
        segmenter_model_type="incremental",
        segmenter_device=args.device,
        segmenter_gpu_batch_size=args.gpu_batch_size,
        segmenter_max_gpu_batch_size=args.max_gpu_batch_size,
        reranker_model_name_or_path=None,
    )
    scorer = segmenter.segmenter_model.model.scorer
    model = scorer.model
    model_device = str(model.device)
    if model_device == "cuda":
        model_device = "cuda:0"
    if model_device != args.device:
        raise RuntimeError(
            f"model resolved to {model_device!r}, expected {args.device!r}"
        )
    try:
        actual_dtype = str(next(model.parameters()).dtype)
    except StopIteration:
        actual_dtype = "unknown"

    runtime = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "minicons_version": package_version("minicons"),
        "hashformers_version": package_version("hashformers"),
        "huggingface_hub_version": package_version("huggingface_hub"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": args.device,
        "resolved_device": model_device,
        "requested_precision": "library-default",
        "actual_parameter_dtype": actual_dtype,
        "quantization": "none",
        "model_commit": spec["revision"],
        "model_snapshot": str(snapshot),
    }
    if model_device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(model_device)
        runtime["cuda_device"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return torch, segmenter, runtime


def segment_once(
    torch: Any,
    segmenter: Any,
    source: str,
    *,
    device: str,
    topk: int,
    steps: int,
) -> tuple[str, float]:
    """Segment one source and return synchronized end-to-end latency."""

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    outputs = segmenter.segment(
        [source],
        topk=topk,
        steps=steps,
        use_reranker=False,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if len(outputs) != 1 or not isinstance(outputs[0], str):
        raise RuntimeError("Hashformers must return one text prediction per input")
    return outputs[0], elapsed_ms


def run_benchmark(args: argparse.Namespace) -> None:
    """Run one isolated Hashformers baseline over its fixed sample scope."""

    manifest_path = args.manifest.resolve()
    complete_manifest = load_jsonl(manifest_path)
    validate_manifest(complete_manifest)
    manifest_sha256 = file_sha256(manifest_path)
    spec = dict(MODEL_SPECS[args.model])
    selected = select_manifest(complete_manifest, spec["scope_dataset"])
    predictions_path, metadata_path = prepare_output_directory(
        args.output_dir.resolve(), args.overwrite
    )

    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evaluation_contract_id": EVALUATION_CONTRACT_ID,
        "status": "loading-model",
        "started_at": utc_now(),
        "repository_revision": git_revision(),
        "repository_dirty": git_dirty(),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "complete_manifest_count": len(complete_manifest),
        "sample_count": len(selected),
        "sample_scope_dataset": spec["scope_dataset"],
        "model": spec,
        "segmentation": {
            "algorithm": "Hashformers beam search",
            "model_type": "incremental",
            "topk": args.topk,
            "steps": args.steps,
            "reranker": None,
            "gpu_batch_size": args.gpu_batch_size,
            "max_gpu_batch_size": args.max_gpu_batch_size,
            "output_contract": "may only insert ASCII spaces into the input",
            "proposal_policy": (
                "strict output, deterministic boundary projection, then source fallback"
            ),
        },
        "measurement": {
            "warmup_items": min(args.warmup, len(selected)),
            "batch_size": 1,
            "cuda_synchronize_before_and_after_segmentation_when_cuda_is_active": True,
            "one_model_per_process": True,
            "latency_scope": "complete TransformerWordSegmenter.segment call",
        },
        "hardware": {
            "platform": platform.platform(),
            "cpu": cpu_metadata(),
            "gpus_from_nvidia_smi": gpu_driver_metadata(),
        },
    }
    write_json(metadata_path, metadata)
    torch, segmenter, runtime = load_segmenter(args, spec)
    metadata["runtime"] = runtime
    runtime_device = runtime["resolved_device"]

    warmup_ids = []
    for sample in selected[: args.warmup]:
        segment_once(
            torch,
            segmenter,
            sample["input"],
            device=runtime_device,
            topk=args.topk,
            steps=args.steps,
        )
        warmup_ids.append(sample["sample_id"])
    if runtime_device.startswith("cuda"):
        torch.cuda.synchronize(runtime_device)
        torch.cuda.reset_peak_memory_stats(runtime_device)
    baseline_memory = memory_snapshot(torch, runtime_device)
    metadata["measurement"]["warmup_sample_ids"] = warmup_ids
    metadata["measurement"]["baseline_gpu_memory"] = baseline_memory
    metadata["status"] = "running"
    write_json(metadata_path, metadata)

    measured_start = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8", newline="\n") as output_handle:
        for sample in selected:
            error = None
            try:
                raw_output, segmentation_ms = segment_once(
                    torch,
                    segmenter,
                    sample["input"],
                    device=runtime_device,
                    topk=args.topk,
                    steps=args.steps,
                )
                (
                    valid,
                    prediction,
                    invalid_reason,
                    output_wrapper,
                    prediction_source,
                    recovery_method,
                ) = propose_segmentation(sample["input"], raw_output)
            except Exception as exc:  # noqa: BLE001
                raw_output = None
                segmentation_ms = None
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
                "evaluation_contract_id": EVALUATION_CONTRACT_ID,
                "manifest_sha256": manifest_sha256,
                **sample,
                "model_label": spec["label"],
                "model_id": spec["model_id"],
                "model_revision": spec["revision"],
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
                # Keep the shared artifact field name; metadata documents that
                # this is full segmenter latency rather than token generation.
                "generation_ms": segmentation_ms,
                "error": error,
            }
            output_handle.write(canonical_json(record) + "\n")
            output_handle.flush()

    if runtime_device.startswith("cuda"):
        torch.cuda.synchronize(runtime_device)
    measured_seconds = time.perf_counter() - measured_start
    completed = load_jsonl(predictions_path)
    runtime_error_count = sum(bool(record.get("error")) for record in completed)
    metadata["status"] = (
        "completed" if runtime_error_count == 0 else "completed-with-errors"
    )
    metadata["completed_at"] = utc_now()
    metadata["measurement"]["measured_wall_seconds"] = measured_seconds
    metadata["measurement"]["throughput_items_per_wall_second"] = (
        len(completed) / measured_seconds if measured_seconds else None
    )
    memory = memory_snapshot(torch, runtime_device)
    memory["baseline_allocated_bytes"] = baseline_memory["baseline_allocated_bytes"]
    metadata["measurement"]["gpu_memory"] = memory
    metadata["measurement"]["candidate_batch_telemetry"] = (
        segmenter.segmenter_model.model.batch_telemetry
    )
    metadata["measurement"]["runtime_error_count"] = runtime_error_count
    metadata["results"] = summarize_records(completed)
    write_json(metadata_path, metadata)
    print(f"saved {len(completed)} predictions to {predictions_path}")
    print(f"saved run metadata and metrics to {metadata_path}")


def stable_id_hash(sample_ids: Sequence[str]) -> str:
    """Hash a sorted shared sample-ID list for paired-comparison auditability."""

    import hashlib

    payload = "\n".join(sorted(sample_ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def compare_runs(runs: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Compare different inference protocols on their shared fixed sample IDs."""

    if len(runs) < 2:
        raise ValueError("comparison requires at least two prediction runs")
    summaries = [summarize_records(run) for run in runs]
    for run in runs:
        if not run:
            raise ValueError("prediction run is empty")
        if run[0]["manifest_sha256"] != runs[0][0]["manifest_sha256"]:
            raise ValueError("all runs must use the same manifest hash")

    comparisons = []
    provenance_fields = (
        "dataset",
        "dataset_revision",
        "split",
        "row_index",
        "group",
        "input",
        "gold",
    )
    for left_index, left in enumerate(runs):
        left_by_id = {str(record["sample_id"]): record for record in left}
        for right in runs[left_index + 1 :]:
            right_by_id = {str(record["sample_id"]): record for record in right}
            shared_ids = sorted(set(left_by_id).intersection(right_by_id))
            if not shared_ids:
                raise ValueError("prediction runs have no shared sample IDs")
            for sample_id in shared_ids:
                for field in provenance_fields:
                    if left_by_id[sample_id].get(field) != right_by_id[sample_id].get(
                        field
                    ):
                        raise ValueError(
                            f"paired sample {sample_id!r} differs in {field}"
                        )
            left_values = [bool(left_by_id[item]["correct"]) for item in shared_ids]
            right_values = [bool(right_by_id[item]["correct"]) for item in shared_ids]
            difference = statistics.fmean(
                int(a) - int(b) for a, b in zip(left_values, right_values)
            )
            low, high = paired_bootstrap_interval(left_values, right_values)
            comparisons.append(
                {
                    "left": left[0]["model_label"],
                    "right": right[0]["model_label"],
                    "left_protocol_id": left[0]["protocol_id"],
                    "right_protocol_id": right[0]["protocol_id"],
                    "evaluation_contract_id": EVALUATION_CONTRACT_ID,
                    "manifest_sha256": left[0]["manifest_sha256"],
                    "left_samples": len(left),
                    "right_samples": len(right),
                    "paired_samples": len(shared_ids),
                    "shared_sample_ids_sha256": stable_id_hash(shared_ids),
                    "accuracy_difference": difference,
                    "ci_95": [low, high],
                    "ci_method": (
                        "paired percentile bootstrap (10,000 resamples, seed 42)"
                    ),
                }
            )
    return {
        "schema_version": 1,
        "evaluation_contract_id": EVALUATION_CONTRACT_ID,
        "manifest_sha256": runs[0][0]["manifest_sha256"],
        "runs": summaries,
        "paired_comparisons": comparisons,
    }


def compare_files(args: argparse.Namespace) -> None:
    """Write summaries and all pairwise shared-sample comparisons."""

    runs = [load_jsonl(path.resolve()) for path in args.predictions]
    comparison = compare_runs(runs)
    comparison["created_at"] = utc_now()
    comparison["sources"] = [
        {"path": str(path.resolve()), "sha256": file_sha256(path.resolve())}
        for path in args.predictions
    ]
    write_json(args.output.resolve(), comparison)
    print(f"saved combined comparison to {args.output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one isolated Hashformers model")
    run.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", type=single_device, default="cuda:0")
    run.add_argument("--gpu-batch-size", type=candidate_batch_size, default="auto")
    run.add_argument("--max-gpu-batch-size", type=int, default=512)
    run.add_argument("--topk", type=int, default=5)
    run.add_argument("--steps", type=int, default=5)
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(handler=run_benchmark)

    compare = subparsers.add_parser(
        "compare", help="compare Qwen and Hashformers prediction artifacts"
    )
    compare.add_argument("--predictions", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(handler=compare_files)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    for field in ("warmup", "topk", "steps", "max_gpu_batch_size"):
        value = getattr(args, field, 1)
        if value < (0 if field == "warmup" else 1):
            raise SystemExit(f"--{field.replace('_', '-')} must be positive")
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
