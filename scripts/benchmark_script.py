#!/usr/bin/env python3
"""
✂️ Hashformers Benchmark: Hugging Face Hub Datasets

This script benchmarks hashformers against various word segmentation approaches
using datasets loaded from the Hugging Face Hub.

Based on the original benchmark_notebook.ipynb but refactored to:
1. Load datasets from Hugging Face Hub instead of local files
2. Organize datasets into 3 groups: English Hashtags, Foreign Hashtags, Identifier Splitting
3. Calculate latency globally across all groups
4. Calculate accuracy metrics separately per group

Categories:
- Classic Statistical: wordninja, symspellpy
- Social Media Specialist: ekphrasis
- Hashformers: gpt2 (baseline), distilgpt2 (no reranker)

Benchmark Data: 20 samples from each dataset on HF Hub
- English Hashtags: boun, stan_small, stan_large, dev_stanford, test_stanford, snap
- Foreign Hashtags: nru_hse, hashset_distant, hashset_manual, hashset_distant_sampled
- Identifier Splitting: loyola, lynx, jhotdraw, binkley, bt11

Requirements: pip install datasets hashformers wordninja symspellpy ekphrasis pandas

This is the archival January 2026 exploratory script. Its Qwen2 adapter
documents the historical configuration but cannot reproduce the published row
because the run did not retain fixed sample IDs, raw outputs, or an exact model
revision. It does not implement the fixed-sample, output-contract, provenance,
statistics, or timing protocol. Use
``scripts/qwen_benchmark.py`` for current prompted-model evaluation.
"""

import os
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Optional
import random
import time

import pandas as pd
import torch
from datasets import load_dataset

random.seed(42)  # For reproducibility

SAMPLES_PER_DATASET = 20

# =============================================================================
# DATASET GROUP DEFINITIONS
# =============================================================================

DATASET_GROUPS = {
    "English Hashtags": [
        "ruanchaves/boun",
        "ruanchaves/stan_small",
        "ruanchaves/stan_large",
        "ruanchaves/dev_stanford",
        "ruanchaves/test_stanford",
        "ruanchaves/snap",
    ],
    "Foreign Hashtags": [
        "ruanchaves/nru_hse",
        "ruanchaves/hashset_distant",
        "ruanchaves/hashset_manual",
        "ruanchaves/hashset_distant_sampled",
    ],
    "Identifier Splitting": [
        "ruanchaves/loyola",
        "ruanchaves/lynx",
        "ruanchaves/jhotdraw",
        "ruanchaves/binkley",
        "ruanchaves/bt11",
    ],
}


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """Install all required packages and download corpora."""
    print("🔧 Setting up environment...")

    # Install packages (commented out - run manually if needed)
    subprocess.run([
         sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/ruanchaves/hashformers.git@benchmark"
     ])
    subprocess.run([
         sys.executable, "-m", "pip", "install", "-q",
         "datasets==3.6.0", "wordninja", "symspellpy", "ekphrasis", "transformers",
         "accelerate", "bitsandbytes", "scipy", "pandas", "matplotlib", "seaborn",
         "git+https://github.com/casics/spiral.git"
     ])

    # Download SymSpell frequency dictionary
    if not os.path.exists("frequency_dictionary_en_82_765.txt"):
        print("Downloading SymSpell dictionary...")
        subprocess.run([
            "curl", "-s", "-o", "frequency_dictionary_en_82_765.txt",
            "https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt"
        ])

    # Clone hashformers repo for datasets if not available
    if not os.path.exists("datasets"):
        print("Cloning hashformers repository for datasets...")
        subprocess.run([
            "git", "clone", "-q", "--depth", "1",
            "https://github.com/ruanchaves/hashformers.git", "temp_repo"
        ])
        subprocess.run(["mv", "temp_repo/datasets", "datasets"])
        subprocess.run(["rm", "-rf", "temp_repo"])

    # Trigger Ekphrasis corpus download
    print("Downloading Ekphrasis corpora...")
    from ekphrasis.classes.preprocessor import TextPreProcessor
    from ekphrasis.classes.tokenizer import SocialTokenizer
    from ekphrasis.dicts.emoticons import emoticons

    # This initialization triggers the download of Twitter corpus files
    _ekphrasis_init = TextPreProcessor(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
        segmenter="twitter",
        corrector="twitter",
        unpack_hashtags=True,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
    )
    del _ekphrasis_init

    print("✅ Environment setup complete!")


setup_environment()

# ============================================================================
# SEGMENTER ARCHITECTURE
# ============================================================================

class Segmenter(ABC):
    """Abstract base class for word segmentation tools."""

    @abstractmethod
    def segment(self, text: str) -> str:
        """
        Segment a hashtag or concatenated string into space-separated words.

        Args:
            text: Input text (hashtag without # symbol)

        Returns:
            Space-separated segmented text
        """
        pass

    def _clean_input(self, text: str) -> str:
        """Remove # symbol and clean input text."""
        return text.lstrip("#").strip()


# ----------------------------------------------------------------------------
# WordNinja Segmenter
# ----------------------------------------------------------------------------
import wordninja


class WordNinjaSegmenter(Segmenter):
    """Word segmentation using WordNinja (statistical n-gram model)."""

    def __init__(self):
        """Initialize WordNinja segmenter."""
        # WordNinja loads its model on first use
        pass

    def segment(self, text: str) -> str:
        """Segment text using WordNinja."""
        cleaned = self._clean_input(text)
        words = wordninja.split(cleaned)
        return " ".join(words)


# ----------------------------------------------------------------------------
# SymSpell Segmenter
# ----------------------------------------------------------------------------
try:
    from symspellpy import SymSpell

    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False


class SymSpellSegmenter(Segmenter):
    """Word segmentation using SymSpell (symmetric delete spelling correction)."""

    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Initialize SymSpell with frequency dictionary.

        Args:
            dictionary_path: Path to the frequency dictionary file.
                            If None, will try to use default from symspellpy package.
        """
        if not SYMSPELL_AVAILABLE:
            raise ImportError("symspellpy is not installed. Run: pip install symspellpy")

        self.sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)

        # Try to load from package resources or specified path
        if dictionary_path:
            if not self.sym_spell.load_dictionary(
                dictionary_path, term_index=0, count_index=1
            ):
                raise FileNotFoundError(f"Dictionary not found: {dictionary_path}")
        else:
            # Try common locations
            try:
                import pkg_resources
                dict_path = pkg_resources.resource_filename(
                    "symspellpy", "frequency_dictionary_en_82_765.txt"
                )
                self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
            except Exception:
                raise FileNotFoundError(
                    "Could not find SymSpell dictionary. "
                    "Please provide dictionary_path or install symspellpy with resources."
                )

    def segment(self, text: str) -> str:
        """Segment text using SymSpell word segmentation."""
        cleaned = self._clean_input(text).lower()
        result = self.sym_spell.word_segmentation(cleaned)
        return result.corrected_string


# ----------------------------------------------------------------------------
# Ekphrasis Segmenter
# ----------------------------------------------------------------------------
try:
    from ekphrasis.classes.preprocessor import TextPreProcessor
    from ekphrasis.classes.tokenizer import SocialTokenizer
    from ekphrasis.dicts.emoticons import emoticons

    EKPHRASIS_AVAILABLE = True
except ImportError:
    EKPHRASIS_AVAILABLE = False


class EkphrasisSegmenter(Segmenter):
    """Word segmentation using Ekphrasis (social media text processor)."""

    def __init__(self, corpus: str = "twitter"):
        """
        Initialize Ekphrasis text processor.

        Args:
            corpus: Corpus for word statistics ('twitter' or 'english')
        """
        if not EKPHRASIS_AVAILABLE:
            raise ImportError("ekphrasis is not installed. Run: pip install ekphrasis")

        self.text_processor = TextPreProcessor(
            normalize=[
                "url",
                "email",
                "percent",
                "money",
                "phone",
                "user",
                "time",
                "date",
                "number",
            ],
            annotate={"hashtag", "allcaps", "elongated", "repeated", "emphasis", "censored"},
            fix_html=True,
            segmenter=corpus,
            corrector=corpus,
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons],
        )

    def segment(self, text: str) -> str:
        """Segment text using Ekphrasis."""
        cleaned = self._clean_input(text)
        # Ekphrasis expects hashtag with # symbol
        tokens = self.text_processor.pre_process_doc("#" + cleaned)
        # Remove the <hashtag> and </hashtag> annotation tokens
        tokens = [t for t in tokens if not t.startswith("<") and not t.endswith(">")]
        return " ".join(tokens)


# ----------------------------------------------------------------------------
# Hashformers Segmenter
# ----------------------------------------------------------------------------
from hashformers import TransformerWordSegmenter


class HashformersSegmenter(Segmenter):
    """Word segmentation using Hashformers (Transformer beam search)."""

    def __init__(
        self,
        segmenter_model: str = "gpt2",
        segmenter_type: str = "incremental",
        reranker_model: Optional[str] = None,
        reranker_type: Optional[str] = None
    ):
        """
        Initialize Hashformers word segmenter.

        Args:
            segmenter_model: HuggingFace model name for segmentation
            segmenter_type: Model type ('incremental', 'masked', or 'seq2seq')
            reranker_model: Optional reranker model name
            reranker_type: Optional reranker model type
        """
        self.ws = TransformerWordSegmenter(
            segmenter_model_name_or_path=segmenter_model,
            segmenter_model_type=segmenter_type,
            reranker_model_name_or_path=reranker_model,
            reranker_model_type=reranker_type
        )
        self.model_name = segmenter_model

    def segment(self, text: str) -> str:
        """Segment text using Hashformers."""
        cleaned = self._clean_input(text)
        results = self.ws.segment([cleaned])
        return results[0] if results else cleaned


# ----------------------------------------------------------------------------
# Spiral Segmenter (using Ronin algorithm)
# ----------------------------------------------------------------------------
try:
    from spiral import ronin

    SPIRAL_AVAILABLE = True
except ImportError:
    SPIRAL_AVAILABLE = False


class SpiralSegmenter(Segmenter):
    """Word segmentation using Spiral (Ronin algorithm)."""

    def __init__(self):
        """
        Initialize Spiral Ronin segmenter.
        """
        if not SPIRAL_AVAILABLE:
            raise ImportError("spiral is not installed. Run: pip install git+https://github.com/casics/spiral.git")

    def segment(self, text: str) -> str:
        """Segment text using Spiral Ronin."""
        cleaned = self._clean_input(text)
        # Spiral Ronin returns a list of tokens
        tokens = ronin.split(cleaned)
        return " ".join(tokens)


# ----------------------------------------------------------------------------
# Historical Local LLM Segmenter (January 2026 Qwen2 configuration)
# ----------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LocalLLMSegmenter(Segmenter):
    """Implement the historical local Qwen2 prompting configuration."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        load_in_4bit: bool = True,
        max_new_tokens: int = 64
    ):
        """
        Initialize local LLM for segmentation.

        Args:
            model_name: HuggingFace model name
            load_in_4bit: Whether to use 4-bit quantization
            max_new_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        # Configure quantization
        if load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def segment(self, text: str) -> str:
        """Segment text using LLM prompting."""
        cleaned = self._clean_input(text)

        # Construct messages for Qwen2 chat format with few-shot examples
        messages = [
            {"role": "system", "content": "You split concatenated hashtag words into separate words. Reply with ONLY the space-separated words, nothing else."},
            # Few-shot examples to guide the model
            {"role": "user", "content": "Split: icecream"},
            {"role": "assistant", "content": "ice cream"},
            {"role": "user", "content": "Split: newyorkcity"},
            {"role": "assistant", "content": "new york city"},
            {"role": "user", "content": "Split: machinelearning"},
            {"role": "assistant", "content": "machine learning"},
            {"role": "user", "content": "Split: throwbackthursday"},
            {"role": "assistant", "content": "throwback thursday"},
            {"role": "user", "content": "Split: gameofthrones"},
            {"role": "assistant", "content": "game of thrones"},
            # Actual query
            {"role": "user", "content": f"Split: {cleaned}"}
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the new tokens (exclude prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean up: take first line, remove quotes and extra whitespace
        answer = answer.split("\n")[0].strip()
        answer = answer.strip('"\'')

        # Remove common prefixes the model might add
        for prefix in ["Answer:", "Result:", "Output:", "Split:"]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()

        # CRITICAL: Only keep characters from the original input + spaces
        # This prevents the LLM from hallucinating new characters
        cleaned_lower = cleaned.lower()
        filtered_chars = []
        input_idx = 0
        for char in answer.lower():
            if char == ' ':
                # Keep spaces (this is the segmentation)
                filtered_chars.append(' ')
            elif input_idx < len(cleaned_lower) and char == cleaned_lower[input_idx]:
                # Keep characters that match the original input in order
                filtered_chars.append(cleaned[input_idx])  # Preserve original case
                input_idx += 1
            # Skip any hallucinated characters

        # If we didn't consume all input characters, the answer is malformed
        if input_idx < len(cleaned):
            # Try a simpler approach: just extract space positions from the answer
            # and insert them into the original input
            answer_no_space = answer.replace(" ", "").lower()
            if answer_no_space == cleaned_lower:
                # Characters match, just different spacing - reconstruct with spaces
                result = []
                orig_idx = 0
                for char in answer.lower():
                    if char == ' ':
                        result.append(' ')
                    else:
                        if orig_idx < len(cleaned):
                            result.append(cleaned[orig_idx])
                            orig_idx += 1
                answer = ''.join(result)
            else:
                # Can't salvage - return original
                return cleaned
        else:
            answer = ''.join(filtered_chars)

        # Clean up multiple spaces
        answer = ' '.join(answer.split())

        # Fallback: if answer is empty or looks wrong, return cleaned input
        if not answer or answer.lower().replace(" ", "") != cleaned.lower():
            return cleaned

        return answer


# ----------------------------------------------------------------------------
# Utility: GPU Memory Management
# ----------------------------------------------------------------------------
import gc


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared. Current allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# =============================================================================
# DATA LOADING FROM HUGGING FACE HUB
# =============================================================================


def get_preferred_split(dataset) -> str:
    """
    Get the preferred split from a dataset following priority: test > validation/dev > train/default.

    Args:
        dataset: A loaded HuggingFace dataset

    Returns:
        The name of the preferred split
    """
    available_splits = list(dataset.keys())

    # Priority order
    priority = ["test", "validation", "dev", "train"]

    for split in priority:
        if split in available_splits:
            return split

    # If none of the priority splits exist, return the first available
    return available_splits[0] if available_splits else None


def load_samples_from_hf(
    dataset_name: str,
    group_name: str,
    n_samples: int = SAMPLES_PER_DATASET,
) -> tuple[list[dict], str]:
    """
    Load samples from a Hugging Face dataset.

    Args:
        dataset_name: Full dataset name on HF Hub (e.g., 'ruanchaves/boun')
        group_name: The group this dataset belongs to
        n_samples: Number of samples to load

    Returns:
        Tuple of (list of sample dictionaries, split used)
    """
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, trust_remote_code=True)

        # Get preferred split
        split_name = get_preferred_split(dataset)
        if split_name is None:
            print(f"  ⚠️ No valid splits found in {dataset_name}")
            return [], "N/A"

        split_data = dataset[split_name]

        # Determine field names based on group
        if group_name == "Identifier Splitting":
            input_field = "identifier"
        else:
            input_field = "hashtag"
        output_field = "segmentation"

        # Sample randomly
        indices = list(range(len(split_data)))
        if len(indices) > n_samples:
            sampled_indices = random.sample(indices, n_samples)
        else:
            sampled_indices = indices

        # Extract samples
        results = []
        for idx in sampled_indices:
            item = split_data[idx]

            # Handle field access
            if input_field in item:
                input_text = str(item[input_field])
            elif "hashtag" in item:
                input_text = str(item["hashtag"])
            elif "identifier" in item:
                input_text = str(item["identifier"])
            else:
                # Try to get the first column
                input_text = str(list(item.values())[0])

            if output_field in item:
                gold = str(item[output_field])
            else:
                # Fallback
                gold = str(list(item.values())[1]) if len(item) > 1 else input_text

            results.append(
                {
                    "input": input_text,
                    "gold": gold,
                    "source": dataset_name.split("/")[-1],
                    "group": group_name,
                }
            )

        return results, split_name

    except Exception as e:
        print(f"  ⚠️ Failed to load {dataset_name}: {e}")
        return [], "ERROR"


def load_all_datasets() -> tuple[pd.DataFrame, dict]:
    """
    Load all datasets from all groups.

    Returns:
        Tuple of (DataFrame with all samples, dict mapping dataset names to splits used)
    """
    all_samples = []
    splits_used = {}

    print("📂 Loading datasets from Hugging Face Hub...")
    print("=" * 60)

    for group_name, datasets in DATASET_GROUPS.items():
        print(f"\n📁 {group_name}:")
        for dataset_name in datasets:
            samples, split = load_samples_from_hf(dataset_name, group_name)
            splits_used[dataset_name] = split
            all_samples.extend(samples)
            print(f"  ✅ {dataset_name.split('/')[-1]}: {len(samples)} samples (split: {split})")

    df = pd.DataFrame(all_samples)
    print()
    print("=" * 60)
    print(f"📊 Total samples loaded: {len(df)}")

    return df, splits_used


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

def run_benchmark(
    segmenters: dict[str, Segmenter],
    dataset: pd.DataFrame,
    input_column: str = "input",
) -> list[dict]:
    """
    Run benchmark across all segmenters on the given dataset.

    Args:
        segmenters: Dictionary mapping model names to Segmenter instances
        dataset: DataFrame containing inputs to segment
        input_column: Column name containing inputs

    Returns:
        List of dictionaries containing benchmark results
    """
    results = []

    inputs = dataset[input_column].tolist()
    groups = dataset["group"].tolist()
    sources = dataset["source"].tolist()

    total_iterations = len(inputs) * len(segmenters)
    print(f"Running benchmark on {len(inputs)} samples with {len(segmenters)} models...")
    print(f"Total iterations: {total_iterations}")

    iteration = 0
    for i, input_text in enumerate(inputs):
        for model_name, segmenter in segmenters.items():
            iteration += 1
            if iteration % 50 == 0:
                print(f"  Progress: {iteration}/{total_iterations} ({100*iteration/total_iterations:.1f}%)")

            try:
                # Measure latency
                start_time = time.perf_counter()
                output = segmenter.segment(input_text)
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                error = None

            except Exception as e:
                output = f"ERROR: {str(e)[:50]}"
                latency_ms = 0.0
                error = str(e)

            results.append(
                {
                    "input": input_text,
                    "model": model_name,
                    "output": output,
                    "latency_ms": latency_ms,
                    "error": error,
                    "group": groups[i],
                    "source": sources[i],
                }
            )

    return results


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert benchmark results to a pandas DataFrame."""
    return pd.DataFrame(results)


def create_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a wide-format comparison table showing outputs side-by-side.

    Args:
        results_df: DataFrame with benchmark results

    Returns:
        Wide-format DataFrame with models as columns
    """
    # Pivot to get outputs side by side
    comparison = results_df.pivot(
        index="input",
        columns="model",
        values="output"
    ).reset_index()

    return comparison




# =============================================================================
# EVALUATION METRICS
# =============================================================================


def evaluate_model_on_group(
    predictions: list[str], gold_truths: list[str]
) -> dict:
    """
    Evaluate predictions against gold standard.

    Args:
        predictions: List of predicted segmentations
        gold_truths: List of gold standard segmentations

    Returns:
        Dictionary with accuracy, precision, recall, and F1 score
    """
    from hashformers.evaluation.modeler import Modeler

    modeler = Modeler()

    for pred, gold in zip(predictions, gold_truths):
        pred_normalized = pred.lower().strip()
        gold_normalized = gold.lower().strip()
        modeler.countEntry(pred_normalized, gold_normalized)

    return {
        "accuracy": modeler.calculateAccuracy(),
        "precision": modeler.calculatePrecision(),
        "recall": modeler.calculateRecall(),
        "f1": modeler.calculateFScore(),
    }


def compute_global_latency(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global latency statistics across all groups.

    Args:
        results_df: DataFrame with benchmark results

    Returns:
        DataFrame with latency statistics per model
    """
    latency_stats = (
        results_df.groupby("model")["latency_ms"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    latency_stats.columns = ["Model", "Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)", "Count"]
    latency_stats = latency_stats.sort_values("Mean (ms)")

    # Calculate throughput
    latency_stats["Throughput (items/sec)"] = latency_stats["Mean (ms)"].apply(
        lambda x: 1000 / x if x > 0 else 0
    )

    return latency_stats


def compute_group_accuracy(
    results_df: pd.DataFrame, benchmark_df: pd.DataFrame, segmenters: dict
) -> dict[str, pd.DataFrame]:
    """
    Compute accuracy metrics separately for each group.

    Args:
        results_df: DataFrame with benchmark results
        benchmark_df: Original benchmark DataFrame with gold truths
        segmenters: Dictionary of segmenter names

    Returns:
        Dictionary mapping group names to accuracy DataFrames
    """
    group_results = {}

    for group_name in DATASET_GROUPS.keys():
        # Filter data for this group
        group_mask = benchmark_df["group"] == group_name
        group_benchmark = benchmark_df[group_mask]
        group_golds = group_benchmark["gold"].tolist()

        if len(group_golds) == 0:
            continue

        evaluation_results = []
        for model_name in segmenters.keys():
            # Get predictions for this model and group
            model_group_results = results_df[
                (results_df["model"] == model_name) & (results_df["group"] == group_name)
            ]
            predictions = model_group_results["output"].tolist()

            if len(predictions) != len(group_golds):
                print(f"  ⚠️ Mismatch for {model_name} in {group_name}: {len(predictions)} vs {len(group_golds)}")
                continue

            # Compute metrics
            metrics = evaluate_model_on_group(predictions, group_golds)
            metrics["model"] = model_name
            evaluation_results.append(metrics)

        if evaluation_results:
            eval_df = pd.DataFrame(evaluation_results)
            eval_df = eval_df[["model", "accuracy", "precision", "recall", "f1"]]
            eval_df = eval_df.sort_values("accuracy", ascending=False)
            group_results[group_name] = eval_df

    return group_results


def analyze_disagreements(comparison_df: pd.DataFrame):
    """Find and display cases where models disagree."""
    # Find cases where models disagree
    def find_disagreements(comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Find hashtags where models produced different outputs."""
        model_cols = [col for col in comparison_df.columns if col != "input"]

        disagreements = []
        for _, row in comparison_df.iterrows():
            outputs = [row[col] for col in model_cols if pd.notna(row[col])]
            # Normalize for comparison (lowercase, strip)
            normalized = [str(o).lower().strip() for o in outputs]
            if len(set(normalized)) > 1:  # Models disagree
                disagreements.append(row)

        return pd.DataFrame(disagreements)

    disagreement_df = find_disagreements(comparison_df)

    print("🔍 Cases Where Models Disagree")
    print("=" * 80)
    print(f"Found {len(disagreement_df)} hashtags with different segmentations")
    print()

    if len(disagreement_df) > 0:
        # Show first 15 disagreements
        print(disagreement_df.head(15).to_string())
    else:
        print("All models produced identical outputs!")

    return disagreement_df


def print_summary(segmenters: dict, benchmark_df: pd.DataFrame, latency_stats: pd.DataFrame):
    """Print final benchmark summary."""
    print("=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)
    print()

    print("🔧 Models Benchmarked:")
    for model in segmenters.keys():
        print(f"   • {model}")
    print()

    print(f"📝 Dataset: {benchmark_df['source'].nunique()} datasets ({len(benchmark_df)} total samples)")
    print()

    print("⏱️ Speed Ranking (fastest to slowest):")
    speed_ranking = latency_stats.sort_values("Mean (ms)")
    for i, (_, row) in enumerate(speed_ranking.iterrows(), 1):
        mean_ms = row["Mean (ms)"]
        if mean_ms > 0:
            throughput = 1000 / mean_ms
            print(".2f")
        else:
            print(f"   {i}. {row['Model']:25s} - {mean_ms:8.2f} ms (N/A - errors occurred)")
    print()

    print("📈 Key Observations:")
    print("   • Statistical methods (WordNinja, SymSpell, Ekphrasis) are significantly faster")
    print("   • Hashformers provides better segmentation quality at the cost of speed")
    print("   • LLM-based segmentation is slowest but can handle complex cases")
    print()

    print("💡 Recommendations:")
    print("   • For high-throughput applications: Use WordNinja or Ekphrasis")
    print("   • For best accuracy: Use Hashformers (consider adding reranker for production)")
    print("   • For research/analysis: Consider the speed-accuracy tradeoff")
    print()

    print("=" * 80)
    print("✅ Benchmark complete! Results saved to benchmark_latency.png")
    print("=" * 80)


def export_results(results_df: pd.DataFrame, comparison_df: pd.DataFrame, latency_stats: pd.DataFrame):
    """Save results to CSV files."""
    # Save detailed results
    results_df.to_csv("benchmark_results_detailed.csv", index=False)
    print("💾 Detailed results saved to: benchmark_results_detailed.csv")

    # Save comparison table
    comparison_df.to_csv("benchmark_comparison.csv", index=False)
    print("💾 Comparison table saved to: benchmark_comparison.csv")

    # Save latency statistics
    latency_stats.to_csv("benchmark_latency_stats.csv", index=False)
    print("💾 Latency statistics saved to: benchmark_latency_stats.csv")

    print()
    print("📁 All results exported successfully!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("✂️ HASHFORMERS BENCHMARK: Hugging Face Hub Datasets")
    print("=" * 80)
    print()
    print("ARCHIVAL RUN: Qwen2 is historical; use scripts/qwen_benchmark.py for Qwen3.")
    print()

    # -------------------------------------------------------------------------
    # 1. Load datasets
    # -------------------------------------------------------------------------
    benchmark_df, splits_used = load_all_datasets()

    if len(benchmark_df) == 0:
        print("❌ No data loaded. Exiting.")
        return

    # -------------------------------------------------------------------------
    # 2. Initialize segmenters
    # -------------------------------------------------------------------------
    print("\n📦 Initializing segmenters...")

    segmenters = {}

    print("  • WordNinja...")
    segmenters["WordNinja"] = WordNinjaSegmenter()

    try:
        print("  • SymSpell...")
        segmenters["SymSpell"] = SymSpellSegmenter()
    except Exception as e:
        print(f"  ⚠️ SymSpell initialization failed: {e}")

    try:
        print("  • Ekphrasis...")
        segmenters["Ekphrasis"] = EkphrasisSegmenter()
    except Exception as e:
        print(f"  ⚠️ Ekphrasis initialization failed: {e}")

    try:
        print("  • Spiral (Ronin)...")
        segmenters["Spiral-Ronin"] = SpiralSegmenter()
    except Exception as e:
        print(f"  ⚠️ Spiral initialization failed: {e}")

    print("  • Hashformers (GPT-2)...")
    segmenters["Hashformers-GPT2"] = HashformersSegmenter(
        segmenter_model="gpt2",
        segmenter_type="incremental",
        reranker_model=None,
        reranker_type=None,
    )

    print("  • Hashformers (DistilGPT2)...")
    segmenters["Hashformers-DistilGPT2"] = HashformersSegmenter(
        segmenter_model="distilgpt2",
        segmenter_type="incremental",
        reranker_model=None,
        reranker_type=None,
    )

    # Clear GPU memory before loading LLM
    clear_gpu_memory()

    # Check if we have enough GPU memory for LLM
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {gpu_mem_gb:.1f} GB")

        if gpu_mem_gb >= 2:
            print("\n  • Historical LLM (Qwen2-0.5B-Instruct, 4-bit quantized)...")
            print("    This may take a few minutes...")

            try:
                segmenters["LLM-Qwen2-Historical"] = LocalLLMSegmenter(
                    model_name="Qwen/Qwen2-0.5B-Instruct",
                    load_in_4bit=True,
                    max_new_tokens=64
                )
                print("    ✅ LLM initialized!")
            except Exception as e:
                print(f"    ⚠️ Could not load LLM: {e}")
                print("    Continuing without LLM segmenter...")
        else:
            print(f"    ⚠️ Insufficient GPU memory ({gpu_mem_gb:.1f} GB). Skipping LLM.")
            print("    Need at least 2 GB for 4-bit quantized Qwen2-0.5B.")
    else:
        print("    ⚠️ No GPU available. Skipping LLM segmenter.")
        print("    Enable GPU runtime for LLM benchmarking")

    print(f"\n✅ Initialized {len(segmenters)} segmenters: {list(segmenters.keys())}")

    # -------------------------------------------------------------------------
    # 3. Run benchmark
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🏃 RUNNING BENCHMARK")
    print("=" * 80)

    results = run_benchmark(
        segmenters=segmenters,
        dataset=benchmark_df,
        input_column="input",
    )

    results_df = pd.DataFrame(results)

    print(f"\n✅ Benchmark complete! Total measurements: {len(results_df)}")

    # Check for errors
    errors = results_df[results_df["error"].notna()]
    if len(errors) > 0:
        print(f"  ⚠️ Errors encountered: {len(errors)}")

    # -------------------------------------------------------------------------
    # 4. Calculate metrics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)

    # 4a. Display splits used
    print("\n📋 Dataset Splits Used:")
    print("-" * 50)
    for dataset_name, split in splits_used.items():
        print(f"  {dataset_name:40s} : {split}")

    # 4b. Global latency
    print("\n⏱️ GLOBAL LATENCY (All Groups Combined)")
    print("-" * 50)
    latency_stats = compute_global_latency(results_df)
    print(latency_stats.to_string(index=False))

    # 4c. Group-specific accuracy
    print("\n📈 ACCURACY BY GROUP")
    print("-" * 50)

    group_accuracy = compute_group_accuracy(results_df, benchmark_df, segmenters)

    for group_name, eval_df in group_accuracy.items():
        print(f"\n🏷️ {group_name}:")
        print(eval_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. Summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📋 FINAL SUMMARY TABLE")
    print("=" * 80)

    # Create combined summary
    print("\n📊 Latency (Global):")
    summary_latency = latency_stats[["Model", "Mean (ms)", "Throughput (items/sec)"]].copy()
    print(summary_latency.to_string(index=False))

    print("\n📊 Accuracy per Group:")
    for group_name, eval_df in group_accuracy.items():
        print(f"\n  {group_name}:")
        for _, row in eval_df.iterrows():
            print(f"    {row['model']:25s} - Acc: {row['accuracy']:6.2f}%, F1: {row['f1']:6.2f}%")

    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
