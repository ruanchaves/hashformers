# Prompted-Qwen segmentation benchmark

This directory defines the reproducible replacement for the prompted-Qwen
portion of the January 2026 benchmark. A complete CUDA run is published under
[`results/2026-08-03-colab-t4-fp16`](results/2026-08-03-colab-t4-fp16/),
including fixed sample IDs, raw generations, run metadata, and the paired
comparison. The archival January numbers remain separate because they used a
different sample and measurement protocol.

## Model scope

The current baseline is
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, loaded with the text-only
`AutoModelForCausalLM` interface and `enable_thinking=False`. Qwen3-0.6B is the
documented fallback, not the newest Qwen model. It was selected because the
Qwen3.5-0.8B multimodal interface would require a different adapter and makes
excluding its vision components from measurement backend-dependent.

`Qwen/Qwen2-0.5B-Instruct` remains available under the
`qwen2-historical` runner key at revision
`c540970f9e29518b1d8f06ab8b24cba66ad77b6d`. This evaluates the historical
model under the refreshed zero-shot, fixed-manifest protocol; it does not
reproduce the January 2026 five-shot NF4 row. That old run did not save sample
IDs or raw outputs and therefore cannot be given a retrospective confidence
interval or invalid-output rate.

## Published fixed-protocol result

Both models were run on August 3, 2026 from repository revision
`b30e66e163bb5ac9d43da23edd725eda7353adf3` in separate processes on one Google
Colab Tesla T4. The checkout was clean for both measurements. The environment
used Python 3.12.13, PyTorch 2.11.0+cu128, Transformers 5.14.1, Accelerate
1.14.0, CUDA runtime 12.8, and NVIDIA driver 580.82.07. Both runs used
unquantized FP16, greedy decoding, batch size one, five warm-up items, CUDA
synchronization, and a 64-token generation ceiling.

| Model | Exact-match accuracy (95% Wilson CI) | Invalid output rate (95% Wilson CI) | Generation latency mean / median / p95 (ms) | Generation throughput (items/s) | Peak allocated / reserved GPU memory (MiB) |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B, non-thinking | 8.57% (5.83%–12.44%) | 66.07% (60.34%–71.37%) | 277.62 / 258.72 / 481.83 | 3.60 | 1,159.39 / 1,606 |
| Qwen2-0.5B-Instruct | 6.43% (4.10%–9.93%) | 86.07% (81.53%–89.64%) | 374.55 / 212.48 / 1,864.39 | 2.67 | 955.63 / 962 |

Wall-clock throughput was 3.59 items/s for Qwen3 and 2.66 items/s for Qwen2.
The paired Qwen3-minus-Qwen2 accuracy difference was 2.14 percentage points
with a 95% paired percentile-bootstrap interval of −1.43 to 5.71 points. The
interval includes zero, so this run does not establish an accuracy difference.
The strict invalid-output contract is part of the measured task: invalid
generations count as incorrect and are not repaired.

These results compare two prompted generative configurations under the same
protocol. They do not compare prompted generation with Hashformers, whose
beam-search configurations were not rerun on this manifest, and they do not
support a claim about LLMs as a class. See the
[`results` README](results/2026-08-03-colab-t4-fp16/README.md) for artifact
checksums and the committed
[`Colab notebook`](issue_78_qwen_benchmark_colab.ipynb) for the GPU workflow.

## Fixed samples

[`samples.jsonl`](samples.jsonl) commits 20 physical rows from each of the 14
datasets that loaded in the January report: 120 English hashtags, 60 non-English
hashtags, and 100 code identifiers. Every record includes the dataset revision,
split, physical row index, input, gold segmentation, and a stable sample ID.
Its SHA-256 is
`743e7519eb4ef760f45a7b5b6a34fea3b0f7394b85e9fed7609b27864cd8497d`.

Rows are selected independently per dataset by ranking physical indices by
`SHA256("42\\0<dataset>\\0<row-index>")`. This avoids global random state and
makes one dataset's size irrelevant to all other samples. The manifest builder
checks every pinned Hugging Face dataset-repository revision and reproduces the
legacy builders from their original sources:

```bash
python3 scripts/build_qwen_sample_manifest.py --output /tmp/samples.jsonl
cmp benchmarks/qwen/samples.jsonl /tmp/samples.jsonl
```

`hashset_manual` remains excluded because its hosted dataset build failed in
the original evaluation. Changing a dataset pin or any fixed record requires a
reviewed benchmark-version change; it is not routine regeneration.

## Inference protocol

Install a CUDA-compatible PyTorch build, then install the benchmark-only
dependencies. Qwen3 requires Transformers 4.51 or newer.

```bash
python -m pip install 'transformers>=4.51,<6' 'accelerate>=1'
```

Run one model on one explicit device per fresh process from a clean checkout.
Use the same precision and quantization for any new model comparison; the
example uses unquantized FP16 on `cuda:0`. Automatic or multi-device placement
is rejected so synchronization and peak-memory measurements identify one GPU.
Do not compare these measurements with the January 2026 Qwen2 NF4 timing.

```bash
python scripts/qwen_benchmark.py run \
  --model qwen3 \
  --manifest benchmarks/qwen/samples.jsonl \
  --device cuda:0 \
  --precision float16 \
  --quantization none \
  --warmup 5 \
  --output-dir benchmark-results/qwen3-fp16

python scripts/qwen_benchmark.py run \
  --model qwen2-historical \
  --manifest benchmarks/qwen/samples.jsonl \
  --device cuda:0 \
  --precision float16 \
  --quantization none \
  --warmup 5 \
  --output-dir benchmark-results/qwen2-fp16

python scripts/qwen_benchmark.py summarize \
  --predictions benchmark-results/qwen3-fp16/predictions.jsonl \
                benchmark-results/qwen2-fp16/predictions.jsonl \
  --output benchmark-results/qwen-comparison.json
```

For NF4, install `bitsandbytes` and pass
`--quantization bnb-4bit-nf4`; the artifact records both the requested
quantization and actual parameter dtype. Keep separately configured runs in
separate result tables.

The prompt is zero-shot and identical across task/language groups. The Qwen3
chat template receives `enable_thinking=False`; decoding is greedy and the
generation settings are recorded. A valid response must reproduce every input
character, with the same case and order, and may insert ASCII spaces only.
Invalid responses are retained verbatim, receive no repaired/fallback
prediction, count as incorrect, and contribute to the reported invalid-output
rate.

Each run saves:

- `predictions.jsonl`: stable sample IDs, raw decoded generations, validated
  predictions, validity reasons, exact-match outcomes, token counts,
  per-item preprocessing/generation timings, protocol/manifest identity, and
  the model precision, quantization, and resolved device;
- `run_metadata.json`: requested and resolved model/tokenizer revisions,
  generation settings, precision/quantization, package versions, OS/CPU/GPU,
  driver/CUDA metadata, manifest and runner hashes, repository revision and
  dirty state, resolved single-device placement, warm-up IDs, throughput, and
  baseline/peak GPU allocation;
- an optional comparison JSON from `summarize`, containing accuracy and
  invalid-rate 95% Wilson intervals plus paired accuracy-difference 95%
  percentile-bootstrap intervals (10,000 resamples, seed 42).
The summarizer refuses paired runs whose protocol, manifest hash, sample IDs,
or per-sample provenance differ.

Latency uses batch size one, five unrecorded warm-up items by default, and
synchronizes the explicitly selected CUDA device immediately before and after
`model.generate`. Tokenization is reported separately. Throughput is reported
both over summed generation time and measured wall time. CUDA model-resident
baseline, peak allocated memory, and peak reserved memory are separate fields.
Model loading is intentionally outside the inference peak.

## Publishing a result

Publish refreshed runs in a separate fixed-protocol section. Do not insert them
into the archival January tables or compare them with historical Hashformers
rows unless those Hashformers configurations are rerun on this exact manifest
under a documented compatible protocol.

Commit or publish the complete prediction JSONL, metadata JSON, and generated
comparison JSON. Verify that all 280 sample IDs occur exactly once, metadata
status is `completed` (not `completed-with-errors`), the manifest hash matches
this file, and the model/tokenizer revisions equal their requested pins. Report
the exact model/configuration and its confidence intervals, invalid-output
rate, latency, throughput, and peak memory; do not generalize the result to
prompted models or LLMs as a class.
