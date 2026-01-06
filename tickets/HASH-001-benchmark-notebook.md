---
name: Feature Request
about: Production-grade benchmark notebook for word segmentation
title: "[Feature] Create comprehensive benchmark Colab notebook comparing hashformers to other segmentation libraries"
labels: enhancement, documentation, benchmark
assignees: ''
---

## 🚀 Feature Request

### Is your feature request related to a problem? Please describe.

Currently, there's no easy way for users to **compare hashformers' performance** against other word segmentation tools. While we have evaluation results in `tutorials/EVALUATION.md` and a script in `scripts/evaluate_ekphrasis.py`, there's no interactive, reproducible notebook that:

- Benchmarks multiple libraries side-by-side
- Measures latency for each approach
- Includes modern LLM-based segmentation for comparison
- Runs easily on Google Colab without local setup

Users often ask: *"How does hashformers compare to wordninja/symspell/ekphrasis?"* and *"Is it worth the extra compute cost?"* We need a definitive answer.

---

### Describe the solution you'd like

A **production-grade Jupyter Notebook** (Google Colab compatible) that benchmarks hashformers against:

| Category | Libraries |
|----------|-----------|
| Classic Statistical | `wordninja`, `symspellpy` |
| Social Media Specialist | `ekphrasis` |
| Modern LLMs | Local quantized model (e.g., `microsoft/Phi-3-mini-4k-instruct`) |
| Hashformers Variants | `gpt2` (baseline) + a more capable HF model that fits on Colab. No reranker |

---

### Proposed Notebook Structure

#### Cell 1: Environment Setup

Single cell with all dependencies:

```bash
!pip install hashformers wordninja symspellpy ekphrasis transformers accelerate bitsandbytes scipy pandas matplotlib seaborn
```

**Critical:** Handle corpus downloads automatically:
- **SymSpell** → Download `frequency_dictionary_en_82_765.txt`
- **Ekphrasis** → Trigger Twitter corpus download via its internal API

---

#### Cell 2: Unified Architecture

Abstract base class with concrete implementations:

```python
from abc import ABC, abstractmethod

class Segmenter(ABC):
    @abstractmethod
    def segment(self, text: str) -> str:
        """Segment concatenated text into space-separated words."""
        pass
```

**Implementations needed:**

| Class | Notes |
|-------|-------|
| `WordNinjaSegmenter` | Simple wrapper around `wordninja.split()` |
| `SymSpellSegmenter` | Load dictionary in `__init__`, use word segmentation lookup |
| `EkphrasisSegmenter` | Configure `TextPreProcessor` with `segmenter="twitter"` |
| `HashformersSegmenter` | Accept `model_name` param, init `TransformerWordSegmenter` |
| `LocalLLMSegmenter` | Use `transformers.pipeline` with 4-bit quantization |

**LLM Prompt:**
```
Split this hashtag into words: {hashtag}
Return only the space-separated words, nothing else.
```

---

#### Cell 3: Benchmark Data

Use datasets already in the repo:

```python
import pandas as pd
df = pd.read_csv("datasets/stan_small.csv")  # ~1100 hashtags with gold truths
```

Available datasets:
- `datasets/stan_small.csv`
- `datasets/stan_large_test.csv`
- `datasets/binkley.csv`

---

#### Cell 4: Execution Engine

```python
def run_benchmark(
    segmenters: dict[str, Segmenter],
    dataset: pd.DataFrame,
    hashtag_column: str = "hashtags",
    limit: int | None = None
) -> list[dict]:
    ...
```

**Metrics per hashtag:**
- `input` — Original hashtag
- `output` — Segmented result
- `model` — Segmenter name
- `latency_ms` — Time via `time.perf_counter()`

---

#### Cell 5: Analysis & Visualization

1. **Comparison Table** — Input vs. all model outputs side-by-side
2. **Latency Bar Chart** — Average ms per model with error bars

```python
import seaborn as sns
sns.barplot(data=results_df, x='model', y='latency_ms', ci='sd')
```

---

### Technical Constraints

**Memory Management (Colab free tier has ~15GB VRAM):**
- Load LLM **last** after other benchmarks complete
- Use 4-bit quantization via `BitsAndBytesConfig(load_in_4bit=True)`
- Provide GPU cache clearing utility:

```python
import torch, gc

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**Code Quality:**
- Type hints throughout
- Docstrings on all classes/functions
- Graceful error handling with informative messages

---

### Acceptance Criteria

- [ ] Notebook runs end-to-end on Google Colab (free tier) without errors
- [ ] All 6 segmenters implemented:
  - [ ] WordNinja
  - [ ] SymSpell  
  - [ ] Ekphrasis
  - [ ] Hashformers (GPT-2)
  - [ ] Hashformers (advanced model)
  - [ ] Local LLM (Phi-3 or similar)
- [ ] SymSpell dictionary auto-downloads
- [ ] Ekphrasis corpus auto-downloads
- [ ] Uses existing datasets from `datasets/` folder
- [ ] Outputs side-by-side comparison table
- [ ] Outputs latency bar chart
- [ ] No OOM errors
- [ ] Type hints throughout
- [ ] All classes have docstrings

---

### Describe alternatives you've considered

1. **Separate benchmark scripts per library** — Harder to compare, not interactive
2. **Static evaluation table in README** — Already exists in `EVALUATION.md`, but not reproducible
3. **HuggingFace Spaces app** — Good for demo, but not for detailed benchmarking

A Colab notebook is the best balance of **reproducibility**, **interactivity**, and **accessibility**.

---

### Additional context

**Related files in repo:**
- `scripts/evaluate_ekphrasis.py` — Existing ekphrasis evaluation (can reference for implementation)
- `tutorials/EVALUATION.md` — Current accuracy/speed tables
- `hashformers.ipynb` — Existing intro notebook (different purpose)
- `datasets/stan_small.csv` — Primary benchmark dataset

**Why include LLMs?**

With the rise of instruction-tuned models, users often ask if prompting an LLM is "good enough" for segmentation. Including a quantized LLM in the benchmark answers this definitively and shows where hashformers' specialized approach excels.

**Sample output table (mockup):**

| Input | WordNinja | SymSpell | Ekphrasis | Hashformers | LLM |
|-------|-----------|----------|-----------|-------------|-----|
| `icecold` | ice cold | ice cold | ice cold | ice cold | ice cold |
| `weneedanationalpark` | we need a national park | wen eed a national park | we need a national park | we need a national park | we need a national park |
| `nobitchassnessfriday` | no bit chass ness friday | no bit chass ness friday | no bitchassness friday | no bitchassness friday | no bitch assness friday |
