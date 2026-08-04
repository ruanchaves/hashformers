# Fixed-manifest Tesla T4 results

This directory contains the complete issue #78 comparison between prompted
Qwen segmentation and Hashformers beam-search segmentation. Every run uses the
same exact-match contract and the fixed manifest with SHA-256
`743e7519eb4ef760f45a7b5b6a34fea3b0f7394b85e9fed7609b27864cd8497d`.
The raw prediction for every evaluated sample is committed.

## Prompted Qwen

The Qwen runs use `hashformers-qwen-space-insertion-v3` and clean repository
revision `59910585795306ca68aefeeba50b30827ae27d12`. Protocol v3 scores a
segmentation proposal for every non-runtime generation without hiding
output-contract failures. Strict insertion-only output is used directly. For
invalid output, a bounded candidate's boundaries may be projected onto the
exact source characters by deterministic edit alignment; if there is no
recoverable boundary signal, the unchanged input is recorded as a source
fallback.

| Model | Proposal accuracy | Strict-output accuracy | Invalid | Recovered | Source fallback |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B, non-thinking | 77/280 (27.50%) | 76/280 (27.14%) | 7/280 (2.50%) | 1/280 (0.36%) | 6/280 (2.14%) |
| Qwen2-0.5B-Instruct | 106/280 (37.86%) | 69/280 (24.64%) | 117/280 (41.79%) | 31/280 (11.07%) | 86/280 (30.71%) |

Of Qwen2's invalid responses, recovery supplied 17 correct proposals and the
explicit source fallback supplied 20. Qwen3 had no correct recovered proposal
and one correct source fallback. Separately, 266/273 strictly valid Qwen3
outputs and 140/163 strictly valid Qwen2 outputs echoed the unchanged input;
only 70/280 gold records need no inserted spaces.

The paired Qwen3-minus-Qwen2 proposal-accuracy difference is −10.36 percentage
points with a 95% paired percentile-bootstrap interval of −14.64 to −6.07
points (10,000 resamples, seed 42).

## Hashformers

The Hashformers runs use `hashformers-beam-search-fixed-manifest-v1` and clean
repository revision `d4180e11e383608387685d8f595103adfae8ee72`. They use top-k
5, five beam-search steps, no reranker, and the PR #80 adaptive candidate batch
controller with `gpu_batch_size="auto"` and a maximum of 512. The controller
selected an effective batch size of 64 for all three models and recorded zero
OOM backoffs. GPT-2 and DistilGPT2 reached `converged`; the short, 20-record
RuGPT3Small scope ended with the controller still marked `tuning`.

| Model | Scope | Exact-match accuracy (95% Wilson CI) | Latency mean / median / p95 (ms) | Throughput (items/s) | Peak allocated / reserved GPU memory (MiB) |
|---|---:|---:|---:|---:|---:|
| Hashformers-GPT2 | 280 | 181/280, 64.64% (58.88%–70.01%) | 95.59 / 71.36 / 268.05 | 10.46 | 1,063.50 / 1,998 |
| Hashformers-DistilGPT2 | 280 | 182/280, 65.00% (59.24%–70.35%) | 70.76 / 51.55 / 162.77 | 14.13 | 824.77 / 1,964 |
| Hashformers-RuGPT3Small | 20 Russian | 17/20, 85.00% (63.96%–94.76%) | 78.90 / 59.09 / 160.17 | 12.67 | 726.60 / 988 |

## Cross-method comparisons

All full-manifest comparisons below are paired over the same 280 sample IDs.
Intervals are 95% paired percentile-bootstrap intervals with 10,000 resamples
and seed 42.

| Difference | Estimate (percentage points) | 95% interval |
|---|---:|---:|
| Qwen3 − Hashformers-GPT2 | −37.14 | −43.21 to −31.07 |
| Qwen3 − Hashformers-DistilGPT2 | −37.50 | −43.57 to −31.07 |
| Qwen2 − Hashformers-GPT2 | −26.79 | −33.21 to −20.36 |
| Qwen2 − Hashformers-DistilGPT2 | −27.14 | −33.21 to −21.07 |
| Hashformers-GPT2 − Hashformers-DistilGPT2 | −0.36 | −3.93 to 3.21 |

On the 20 Russian records only, both Qwen configurations minus RuGPT3Small
were −30 points with a −55 to −5 point interval. That small language-specific
comparison is less precise and must not be treated as a 280-record result.

These results support a scoped comparison of these specialized beam-search and
prompted-generative configurations. They do not establish a conclusion about
LLMs generally. Qwen generation latency and Hashformers segmentation latency
cover different inference paths; their values are reported for reproducibility,
not as an architecture-independent speed claim.

## Hosted provider projection

[`hosted-api-cost-projection.svg`](hosted-api-cost-projection.svg) projects
batch time, total cost, and quality-adjusted cost for
Hashformers-DistilGPT2 on a rented T4 versus representative hosted API prices.
This figure does not use Qwen. The Hashformers input is the measured 14.11
items/s wall throughput and 65.00% exact-match accuracy from the run in this
directory. API cost uses the standard list prices and token profile recorded in
[`hosted_api_cost_scenario.json`](../../hosted_api_cost_scenario.json).

The default scenario assumes 10 concurrent 100-item API requests, 0.5 seconds
of fixed request latency, 100 output tokens/s/request, and 90% hosted-provider
exact-match accuracy. API latency and accuracy are explicitly hypothetical;
they were not measured on this manifest. Hashformers cost uses $0.35 per T4
accelerator-hour with a 60-second minimum and excludes all VM and operational
costs.

| Hosted API price scenario | Projected cost / 1M hashtags | First volume where fresh-job Hashformers spend is lower | First volume where Hashformers cost / expected correct is lower |
|---|---:|---:|---:|
| OpenAI GPT-5.6 Terra | $198.99 | 30 | 41 |
| Anthropic Claude Haiku 4.5 | $68.13 | 86 | 119 |
| Google Gemini 3 Flash Preview | $39.80 | 147 | 203 |
| Hashformers-DistilGPT2 on one T4 | $6.89 | — | — |

The API time scenario first becomes faster at 169 hashtags because it assumes
parallel requests. At one million hashtags it projects 3.32 hours for the API
scenario versus 19.69 hours on one T4. The machine-readable calculations and
caveats are in
[`hosted-api-cost-projection.json`](hosted-api-cost-projection.json), and the
projection can be regenerated with `scripts/segmentation_cost_projection.py`.

## Environment

All runs used one Google Colab Tesla T4. The Qwen environment used Python
3.12.13, PyTorch 2.11.0+cu128, Transformers 5.14.1, Accelerate 1.14.0, CUDA
runtime 12.8, NVIDIA driver 580.82.07, unquantized FP16, greedy decoding, batch
size one, and five warm-up items. The Hashformers metadata records its exact
package versions, pinned model snapshots, adaptive-batch telemetry, warm-up
IDs, synchronized timings, and GPU memory measurements. Every run completed
its intended sample scope with zero runtime errors.

## SHA-256 checksums

```text
f0a32b8c8a65995a1202bd10e693fa6b4cfb97b4a46dbed830c848ff835e0cd4  ../../hosted_api_cost_scenario.json
85a495529cfdd63de6ee85689aa26ff4aa21434ad86a8ade76c9953bebcf871f  hosted-api-cost-projection.json
3379419b23c07e5a88b2252344a0cca206c4e5430c278d645d65efe3fb9b390d  hosted-api-cost-projection.svg
9bcb65487c20498cb2ad1960fefaf76767c269fa2b1a8de1fd37ac509deee5ac  comparison.json
ce4abae9afdc7a601c3ee3669a7c4967467ca63f4e62a50b94ff2294e935d8dc  combined_comparison.json
fa5706930701d655aded0c929117cb4c4a1e43b428b0b9b0d227256311e9e9a9  qwen2/predictions.jsonl
92f07ae72d9317b6e711067a2ea371ac989b3952ed2a8a039061971d0b840e38  qwen2/run_metadata.json
36aa5af63572a2056d04088b3135d015980f6ccb2d59ecc7d7206fbdfee3e836  qwen3/predictions.jsonl
d1537b240208b5f167e84c583dc9242d7cbe7139ad14ed103cc29ea8da426767  qwen3/run_metadata.json
485db86dc5ed01da2851410978afaf5f906df50f0c5f57a1196727600e03f153  hashformers-gpt2/predictions.jsonl
8e1f576670aeec11ce0d4d861ac51d095a0b0c359565f12873ce2c2aea007889  hashformers-gpt2/run_metadata.json
906b85d23cb40a357b6cc306a5b3b1f3a37607d7bfe0a11425b38e28706e9f49  hashformers-distilgpt2/predictions.jsonl
eb2b69920cd7abb783615e34b755b4a3038a6c2e8d5ab2f75d897a9bede5277c  hashformers-distilgpt2/run_metadata.json
fb1f4650b3ae64e7dd94663518ce7cf7e3c76d060a69907f5c58512ecceb00da  hashformers-rugpt3small/predictions.jsonl
3cce7d8714ad29418ad3a56aea78e794dde8f5fda03ea6bcc9ac5cbceeffe48e  hashformers-rugpt3small/run_metadata.json
```

The downloaded Qwen archive was `hashformers-issue-78-v3-results.zip` with
SHA-256 `60c12e00580e4194689f443a405434fa4479b59341877c55dd7dfd8227ee9964`.
The downloaded adaptive Hashformers archive was
`hashformers-issue-78-auto-baselines.zip` with SHA-256
`17bbb3e7a752233177b4f232f300b2844fdaaeb3bc1c4afce4c0617f6728ffa4`.
