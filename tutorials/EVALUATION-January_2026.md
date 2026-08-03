# Hashformers Evaluation Report — January 2026

> **Benchmark Overview:** This report evaluates the performance of various text segmentation approaches across English hashtags, foreign hashtags, and code identifier splitting tasks.

> [!CAUTION]
> This is an archival report. The prompted-model row evaluates only one
> five-shot, 4-bit NF4 `Qwen/Qwen2-0.5B-Instruct` configuration. The run did not
> retain sample IDs or raw generations, did not measure invalid-output rates,
> and did not calculate confidence intervals. Its accuracy and timing values
> are descriptive historical results, not a general comparison with LLMs.

The original scripts are available in the [scripts](https://github.com/ruanchaves/hashformers/tree/main/scripts)
directory (`scripts/benchmark_script_focused.py` and
`scripts/benchmark_script.py`). A new pinned
[Qwen3 protocol](../benchmarks/qwen/README.md) fixes the sampling, output,
provenance, statistical, and performance-measurement issues. It uses
Qwen3-0.6B as the documented text-only, non-thinking fallback for Qwen3.5-0.8B.
No refreshed model result is reported until its raw artifacts are published.

---

## 📋 Datasets Used

| Dataset | Split Used |
|---------|------------|
| [ruanchaves/boun](https://huggingface.co/datasets/ruanchaves/boun) | test |
| [ruanchaves/stan_small](https://huggingface.co/datasets/ruanchaves/stan_small) | test |
| [ruanchaves/stan_large](https://huggingface.co/datasets/ruanchaves/stan_large) | test |
| [ruanchaves/dev_stanford](https://huggingface.co/datasets/ruanchaves/dev_stanford) | validation |
| [ruanchaves/test_stanford](https://huggingface.co/datasets/ruanchaves/test_stanford) | test |
| [ruanchaves/snap](https://huggingface.co/datasets/ruanchaves/snap) | train |
| [ruanchaves/nru_hse](https://huggingface.co/datasets/ruanchaves/nru_hse) | test |
| [ruanchaves/hashset_distant](https://huggingface.co/datasets/ruanchaves/hashset_distant) | test |
| [ruanchaves/hashset_distant_sampled](https://huggingface.co/datasets/ruanchaves/hashset_distant_sampled) | test |
| [ruanchaves/loyola](https://huggingface.co/datasets/ruanchaves/loyola) | test |
| [ruanchaves/lynx](https://huggingface.co/datasets/ruanchaves/lynx) | test |
| [ruanchaves/jhotdraw](https://huggingface.co/datasets/ruanchaves/jhotdraw) | test |
| [ruanchaves/binkley](https://huggingface.co/datasets/ruanchaves/binkley) | test |
| [ruanchaves/bt11](https://huggingface.co/datasets/ruanchaves/bt11) | test |

> [!NOTE]
> `ruanchaves/hashset_manual` was excluded due to a loading error.

---

## ⏱️ Historical Global Latency Performance

| Model | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | Throughput (items/sec) |
|-------|----------:|---------:|---------:|---------:|-----------------------:|
| WordNinja | 0.19 | 0.14 | 0.02 | 2.05 | **5,357** |
| SymSpell | 0.28 | 0.21 | 0.04 | 1.49 | 3,580 |
| Ekphrasis | 0.69 | 0.97 | 0.11 | 9.17 | 1,449 |
| Hashformers-DistilGPT2 | 264.11 | 320.63 | 13.56 | 3,124.96 | 3.79 |
| LLM-Qwen2 (0.5B, historical NF4 run) | 300.63 | 166.81 | 134.10 | 2,444.61 | 3.33 |
| Hashformers-GPT2 | 362.97 | 424.75 | 22.04 | 3,644.44 | 2.76 |

> [!WARNING]
> These latency values used one-item calls without a documented warm-up, CUDA
> synchronization, isolated model processes, common precision, or peak-memory
> measurement. They describe this run only and should not be used for a current
> hardware or architecture comparison.

---

## 📈 Accuracy Results by Task

### English Hashtags

| Rank | Model | Accuracy | Precision | Recall | F1 |
|:----:|-------|----------:|----------:|-------:|---:|
| 🥇 | **Hashformers-DistilGPT2** | **76.67%** | 85.34% | 78.88% | **81.99%** |
| 🥈 | Hashformers-GPT2 | 75.83% | 85.27% | 76.10% | 80.42% |
| 🥉 | Ekphrasis | 72.50% | 77.39% | 80.48% | 78.91% |
| 4 | WordNinja | 71.67% | 71.48% | 84.86% | 77.60% |
| 5 | SymSpell | 69.17% | 69.31% | 80.08% | 74.31% |
| 6 | LLM-Qwen2 (0.5B) | 68.33% | 75.45% | 67.33% | 71.16% |

### Foreign (Non-English) Hashtags

| Rank | Model | Accuracy | Precision | Recall | F1 |
|:----:|-------|----------:|----------:|-------:|---:|
| 🥇 | **Ekphrasis** | **83.33%** | 92.19% | 84.29% | **88.06%** |
| 🥈 | Hashformers-GPT2 | 78.33% | 86.03% | 83.57% | 84.78% |
| 🥉 | Hashformers-DistilGPT2 | 76.67% | 83.94% | 82.14% | 83.03% |
| 4 | LLM-Qwen2 (0.5B) | 60.00% | 75.45% | 59.29% | 66.40% |
| 5 | SymSpell | 48.33% | 46.67% | 55.00% | 50.49% |
| 6 | WordNinja | 35.00% | 48.73% | 55.00% | 51.68% |

### Code Identifier Splitting

| Rank | Model | Accuracy | Precision | Recall | F1 |
|:----:|-------|----------:|----------:|-------:|---:|
| 🥇 | **Ekphrasis** | **66.00%** | 84.84% | 80.76% | **82.75%** |
| 🥈 | WordNinja | 60.00% | 74.57% | 74.57% | 74.57% |
| 🥉 | SymSpell | 60.00% | 69.93% | 68.73% | 69.32% |
| 4 | LLM-Qwen2 (0.5B) | 49.00% | 70.47% | 46.74% | 56.20% |
| 5 | Hashformers-GPT2 | 43.00% | 63.74% | 37.46% | 47.19% |
| 6 | Hashformers-DistilGPT2 | 38.00% | 58.62% | 35.05% | 43.87% |

---

## 🇷🇺 Russian Language Benchmark (NRU HSE Dataset)

> This supplementary benchmark evaluates Russian hashtag segmentation using a language-specific model.

| Rank | Model | Accuracy | Precision | Recall | F1 |
|:----:|-------|----------:|----------:|-------:|---:|
| 🥇 | **Hashformers-RuGPT3Small** | **80.00%** | 83.87% | 81.25% | **82.54%** |
| 🥈 | Hashformers-GPT2 | 75.00% | 75.00% | 75.00% | 75.00% |
| 🥉 | Hashformers-DistilGPT2 | 70.00% | 74.19% | 71.88% | 73.02% |
| 4 | Ekphrasis | 50.00% | 56.00% | 43.75% | 49.12% |
| 5 | SymSpell | 45.00% | 45.00% | 28.13% | 34.62% |
| 5 | LLM-Qwen2 (0.5B) | 45.00% | 45.00% | 28.13% | 34.62% |
| 7 | WordNinja | 0.00% | 10.00% | 6.25% | 7.69% |

> [!IMPORTANT]
> On these 20 sampled records, **RuGPT3Small** scored 5 percentage points above
> GPT-2 and 10 points above DistilGPT2. This small historical result does not by
> itself establish a causal or population-level improvement.

---

## 📊 Summary: Overall Performance Comparison

```
                          English    Foreign      Code     Russian
                         Hashtags   Hashtags Identifiers  Hashtags
                         ────────   ────────  ──────────  ────────
Hashformers-DistilGPT2     🥇         🥉         6th        🥉
Hashformers-GPT2           🥈         🥈         5th        🥈
Hashformers-RuGPT3Small     —          —          —         🥇
Ekphrasis                  🥉         🥇         🥇          4th
WordNinja                  4th        6th        🥈          7th
SymSpell                   5th        5th        🥉          5th
LLM-Qwen2 (0.5B)           6th        4th        4th         5th
```

---

## 🎯 Conclusions

### Key Findings

1. **Hashformers-DistilGPT2 had the highest observed English exact-match accuracy in this sample** (76.67%). Its observed difference from the historical Qwen2-0.5B configuration was 8.34 percentage points (10 of 120 predictions). Because raw paired predictions were not saved, the report cannot attach a paired confidence interval or significance test to that difference.

2. **Heuristic methods led on this code-identifier sample.** Ekphrasis had 66% observed accuracy, while the two Hashformers configurations had lower observed accuracy. The experiment does not isolate whether pretraining data, tokenization, search settings, or another factor caused the difference.

3. **The language-specific backbone had the highest observed Russian result.** Hashformers-RuGPT3Small reached 80% on the 20 sampled NRU HSE records. This small result is consistent with a benefit from matching the backbone to the language, but it is not a precise estimate of that benefit.

4. **This run showed a latency/accuracy trade-off, but its timing protocol was insufficient for a hardware-independent conclusion.** Deployment choices should be benchmarked on the target workload with warm-up, synchronization, throughput, precision, and memory reported.

5. **The useful comparison is specialized beam-search segmentation versus one prompted generative configuration.** Qwen2-0.5B has substantially more parameters than the GPT-2 and DistilGPT2 backbones, and both approaches use Transformer-family language models. This experiment does not show that Hashformers outperforms similarly sized models or LLMs generally.

---

## When to Use Hashformers?

The table below records practical considerations rather than conclusions about
an entire model class.

| Approach | Examples | Recommended When... | Notes |
|----------|----------|---------------------|-------|
| **Heuristic-based** | [SymSpell](https://github.com/wolfgarbe/SymSpell), [Ekphrasis](https://github.com/cbaziotis/ekphrasis), [WordNinja](https://github.com/keredson/wordninja), [Spiral (Ronin)](https://github.com/casics/spiral) | • **Scalability** is a primary requirement.<br><br>• The segmentation domain works well with a standard pre-built vocabulary. | Fast and efficient, but requires a pre-built vocabulary which can be limiting for niche domains or languages. |
| **Hashformers** | [Hashformers](https://github.com/ruanchaves/hashformers) | • You want beam-search segmentation backed by a language model.<br><br>• An appropriate domain/language backbone is available, while a manual vocabulary is not. | Results depend on the backbone, language, dataset, and search configuration. |
| **Prompted generative segmentation** | [Pinned Qwen protocol](../benchmarks/qwen/README.md) | • You want to evaluate direct generation under an insertion-only contract.<br><br>• Invalid outputs, generation latency, and memory are explicitly measured. | The historical row covers one Qwen2-0.5B prompt/configuration only. A current Qwen3 fallback run is pending publication of raw artifacts. |

---

## Appendix: Models Evaluated

| Category | Model | Description |
|----------|-------|-------------|
| **Heuristic** | [WordNinja](https://github.com/keredson/wordninja) | Statistical word segmentation based on Wikipedia unigram frequencies |
| **Heuristic** | [SymSpell](https://github.com/wolfgarbe/SymSpell) | Fast spelling correction and word segmentation using Symmetric Delete |
| **Heuristic** | [Ekphrasis](https://github.com/cbaziotis/ekphrasis) | Text preprocessing tool optimized for social media text |
| **Heuristic** | [Spiral-Ronin](https://github.com/casics/spiral) | Identifier splitting for source code analysis |
| **Hashformers** | [Hashformers-GPT2](https://github.com/ruanchaves/hashformers) | GPT-2 backbone with specialized hashtag segmentation head |
| **Hashformers** | [Hashformers-DistilGPT2](https://github.com/ruanchaves/hashformers) | Smaller GPT-2 variant used by the historical beam-search configuration |
| **Hashformers** | [Hashformers-RuGPT3Small](https://github.com/ruanchaves/hashformers) | Russian-language GPT-3 backbone for Cyrillic text: [ai-forever/rugpt3small_based_on_gpt2](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2) |
| **Prompted generative (historical)** | [LLM-Qwen2 (0.5B)](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | Five-shot, 4-bit NF4 `Qwen/Qwen2-0.5B-Instruct`; raw outputs, fixed IDs, confidence intervals, and invalid-output rates were not retained. |
| **Prompted generative (refresh pending)** | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Pinned text-only Qwen3 fallback with `enable_thinking=False`; see the [reproducible protocol](../benchmarks/qwen/README.md). No result is claimed yet. |

---

*Report generated: January 2026. Methodology caveats and refresh protocol added August 2026.*
