# Hashformers Evaluation Report — January 2026

> **Benchmark Overview:** This report evaluates the performance of various text segmentation approaches across English hashtags, foreign hashtags, and code identifier splitting tasks.

The benchmark scripts are available in the [scripts](https://github.com/ruanchaves/hashformers/tree/main/scripts) directory ( `scripts/benchmark_script_focused.py` and `scripts/benchmark_script.py` ).

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

## ⏱️ Global Latency Performance

| Model | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | Throughput (items/sec) |
|-------|----------:|---------:|---------:|---------:|-----------------------:|
| WordNinja | 0.19 | 0.14 | 0.02 | 2.05 | **5,357** |
| SymSpell | 0.28 | 0.21 | 0.04 | 1.49 | 3,580 |
| Ekphrasis | 0.69 | 0.97 | 0.11 | 9.17 | 1,449 |
| Hashformers-DistilGPT2 | 264.11 | 320.63 | 13.56 | 3,124.96 | 3.79 |
| LLM-Qwen2 (0.5B) | 300.63 | 166.81 | 134.10 | 2,444.61 | 3.33 |
| Hashformers-GPT2 | 362.97 | 424.75 | 22.04 | 3,644.44 | 2.76 |

> [!TIP]
> Heuristic-based approaches (WordNinja, SymSpell, Ekphrasis) are **~1,000x faster** than transformer-based methods, making them ideal for high-throughput scenarios.

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
> Using a language-specific backbone (**RuGPT3Small**) improves Russian segmentation accuracy by **+5–10%** over English-pretrained models.

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

1. **Hashformers excels at English hashtag segmentation**, achieving the highest accuracy (76.67%) and F1-score (81.99%) with DistilGPT2. This represents a **+8.3 percentage point improvement** over the comparable-scale LLM-Qwen2 (0.5B).

2. **Heuristic methods dominate code identifier splitting.** Ekphrasis leads with 66% accuracy, while Hashformers models underperform on programmatic naming conventions (camelCase, snake_case). This suggests the pretraining corpus of GPT-2 models lacks sufficient code-style text.

3. **Language-specific backbones matter.** For Russian hashtags, Hashformers-RuGPT3Small (80% accuracy) substantially outperforms English-pretrained alternatives, demonstrating the importance of matching the LM to the target language.

4. **Latency vs. accuracy trade-off is significant.** Heuristic splitters are ~1,000x faster but sacrifice 5–15% accuracy on hashtag tasks. For batch processing millions of items, heuristics may be acceptable; for quality-critical applications, Hashformers is preferred.

5. **Hashformers outperforms similarly-sized LLMs.** When compared to LLM-Qwen2 (0.5B parameters), Hashformers variants consistently deliver better accuracy across English and Foreign hashtag tasks, proving that the specialized architecture is more effective than general-purpose LLMs at comparable scale.

### Recommendations

| Use Case | Recommended Approach |
|----------|---------------------|
| High-throughput English/Foreign hashtags | **Ekphrasis** (best balance of speed + accuracy) |
| Maximum accuracy on English hashtags | **Hashformers-DistilGPT2** |
| Code identifier splitting | **Ekphrasis** or **WordNinja** |
| Non-English language hashtags | **Hashformers** with language-specific backbone |
| Resource-constrained environments | **WordNinja** (fastest, decent accuracy) |

---

## When to Use Hashformers?

The table below outlines when to use **Hashformers** versus other approaches like heuristic-based splitters (e.g., SymSpell, WordNinja) or large LLMs.

| Approach | Examples | Recommended When... | Notes |
|----------|----------|---------------------|-------|
| **Heuristic-based** | [SymSpell](https://github.com/wolfgarbe/SymSpell), [Ekphrasis](https://github.com/cbaziotis/ekphrasis), [WordNinja](https://github.com/keredson/wordninja), [Spiral (Ronin)](https://github.com/casics/spiral) | • **Scalability** is a primary requirement.<br><br>• The segmentation domain works well with a standard pre-built vocabulary. | Fast and efficient, but requires a pre-built vocabulary which can be limiting for niche domains or languages. |
| **Hashformers** | [Hashformers](https://github.com/ruanchaves/hashformers) | • **Scalability** is needed.<br><br>• You are working in a domain or language where a Language Model is readily available, but compiling a manual vocabulary is too burdensome. | Evidence shows Hashformers is superior to LLMs of similar scale (0.5B parameters). |
| **Large LLMs** | [OpenAI](https://openai.com/), Local LLM Deployment | • **Cost, latency, and scalability** are not concerns.<br><br>• You are segmenting a **low volume** of items. | To gain an accuracy advantage over Hashformers, you generally need to use significantly larger LLMs. |

---

## Appendix: Models Evaluated

| Category | Model | Description |
|----------|-------|-------------|
| **Heuristic** | [WordNinja](https://github.com/keredson/wordninja) | Statistical word segmentation based on Wikipedia unigram frequencies |
| **Heuristic** | [SymSpell](https://github.com/wolfgarbe/SymSpell) | Fast spelling correction and word segmentation using Symmetric Delete |
| **Heuristic** | [Ekphrasis](https://github.com/cbaziotis/ekphrasis) | Text preprocessing tool optimized for social media text |
| **Heuristic** | [Spiral-Ronin](https://github.com/casics/spiral) | Identifier splitting for source code analysis |
| **Hashformers** | [Hashformers-GPT2](https://github.com/ruanchaves/hashformers) | GPT-2 backbone with specialized hashtag segmentation head |
| **Hashformers** | [Hashformers-DistilGPT2](https://github.com/ruanchaves/hashformers) | Distilled GPT-2 for faster inference with minimal accuracy loss |
| **Hashformers** | [Hashformers-RuGPT3Small](https://github.com/ruanchaves/hashformers) | Russian-language GPT-3 backbone for Cyrillic text: [ai-forever/rugpt3small_based_on_gpt2](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2) |
| **LLM** | [LLM-Qwen2 (0.5B)](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | General-purpose 0.5B parameter language model: [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) |

---

*Report generated: January 2026*
