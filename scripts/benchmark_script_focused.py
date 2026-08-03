#!/usr/bin/env python3
"""
✂️ Hashformers Focused Benchmark

This script performs two focused benchmark evaluations:
1. microsoft/CodeGPT-small-py on Identifier Splitting datasets only
2. All current models + ai-forever/rugpt3small_based_on_gpt2 on ruanchaves/nru_hse

Based on the original benchmark_script.py but focused on specific evaluations.

Requirements: pip install datasets hashformers wordninja symspellpy ekphrasis pandas

This is the archival January 2026 focused script. Its Qwen2 adapter documents
the historical configuration but cannot reproduce the published row because
fixed sample IDs, raw outputs, and the exact model revision were not retained.
Use ``scripts/qwen_benchmark.py`` for the fixed-sample, auditable Qwen3 fallback
protocol.
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
# DATASET GROUP DEFINITIONS (FOCUSED)
# =============================================================================

IDENTIFIER_SPLITTING_DATASETS = [
    "ruanchaves/loyola",
    "ruanchaves/lynx",
    "ruanchaves/jhotdraw",
    "ruanchaves/binkley",
    "ruanchaves/bt11",
]

NRU_HSE_DATASET = "ruanchaves/nru_hse"


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
# DATA LOADING FROM HUGGING FACE HUB (FOCUSED)
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


def load_identifier_splitting_datasets() -> tuple[pd.DataFrame, dict]:
    """
    Load all Identifier Splitting datasets.

    Returns:
        Tuple of (DataFrame with all samples, dict mapping dataset names to splits used)
    """
    all_samples = []
    splits_used = {}

    print("📂 Loading Identifier Splitting datasets from Hugging Face Hub...")
    print("=" * 60)

    for dataset_name in IDENTIFIER_SPLITTING_DATASETS:
        samples, split = load_samples_from_hf(dataset_name, "Identifier Splitting")
        splits_used[dataset_name] = split
        all_samples.extend(samples)
        print(f"  ✅ {dataset_name.split('/')[-1]}: {len(samples)} samples (split: {split})")

    df = pd.DataFrame(all_samples)
    print()
    print("=" * 60)
    print(f"📊 Total identifier samples loaded: {len(df)}")

    return df, splits_used


def load_nru_hse_dataset() -> tuple[pd.DataFrame, str]:
    """
    Load samples from ruanchaves/nru_hse dataset.

    Returns:
        Tuple of (DataFrame with samples, split used)
    """
    print("📂 Loading NRU HSE dataset from Hugging Face Hub...")
    print("=" * 60)

    samples, split = load_samples_from_hf(NRU_HSE_DATASET, "Foreign Hashtags")
    df = pd.DataFrame(samples)

    print(f"  ✅ {NRU_HSE_DATASET.split('/')[-1]}: {len(samples)} samples (split: {split})")
    print()
    print("=" * 60)
    print(f"📊 Total NRU HSE samples loaded: {len(df)}")

    return df, split


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


def compute_accuracy(
    results_df: pd.DataFrame, benchmark_df: pd.DataFrame, segmenters: dict
) -> pd.DataFrame:
    """
    Compute accuracy metrics for the dataset.

    Args:
        results_df: DataFrame with benchmark results
        benchmark_df: Original benchmark DataFrame with gold truths
        segmenters: Dictionary of segmenter names

    Returns:
        DataFrame with accuracy metrics per model
    """
    evaluation_results = []
    group_golds = benchmark_df["gold"].tolist()

    for model_name in segmenters.keys():
        # Get predictions for this model
        model_results = results_df[results_df["model"] == model_name]
        predictions = model_results["output"].tolist()

        if len(predictions) != len(group_golds):
            print(f"  ⚠️ Mismatch for {model_name}: {len(predictions)} vs {len(group_golds)}")
            continue

        # Compute metrics
        metrics = evaluate_model_on_group(predictions, group_golds)
        metrics["model"] = model_name
        evaluation_results.append(metrics)

    eval_df = pd.DataFrame(evaluation_results)
    eval_df = eval_df[["model", "accuracy", "precision", "recall", "f1"]]
    eval_df = eval_df.sort_values("accuracy", ascending=False)

    return eval_df


def export_results(results_df: pd.DataFrame, comparison_df: pd.DataFrame, latency_stats: pd.DataFrame, suffix: str):
    """Save results to CSV files."""
    # Save detailed results
    results_df.to_csv(f"benchmark_results_detailed_{suffix}.csv", index=False)
    print(f"💾 Detailed results saved to: benchmark_results_detailed_{suffix}.csv")

    # Save comparison table
    comparison_df.to_csv(f"benchmark_comparison_{suffix}.csv", index=False)
    print(f"💾 Comparison table saved to: benchmark_comparison_{suffix}.csv")

    # Save latency statistics
    latency_stats.to_csv(f"benchmark_latency_stats_{suffix}.csv", index=False)
    print(f"💾 Latency statistics saved to: benchmark_latency_stats_{suffix}.csv")

    print()


def print_evaluation_summary(segmenters: dict, benchmark_df: pd.DataFrame, latency_stats: pd.DataFrame, eval_df: pd.DataFrame, title: str):
    """Print evaluation summary."""
    print("=" * 80)
    print(f"📊 {title}")
    print("=" * 80)
    print()

    print("🔧 Models Evaluated:")
    for model in segmenters.keys():
        print(f"   • {model}")
    print()

    print(f"📝 Dataset: {len(benchmark_df)} samples from {benchmark_df['source'].nunique()} sources")
    print()

    print("⏱️ Performance Ranking (fastest to slowest):")
    speed_ranking = latency_stats.sort_values("Mean (ms)")
    for i, (_, row) in enumerate(speed_ranking.iterrows(), 1):
        mean_ms = row["Mean (ms)"]
        if mean_ms > 0:
            throughput = 1000 / mean_ms
            print(".2f")
        else:
            print(f"   {i}. {row['Model']:25s} - {mean_ms:8.2f} ms (N/A - errors occurred)")
    print()

    print("📈 Accuracy Results:")
    print(eval_df.to_string(index=False))
    print()

    print("=" * 80)
    print("✅ Evaluation complete!")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("✂️ HASHFORMERS FOCUSED BENCHMARK")
    print("=" * 80)
    print()
    print("ARCHIVAL RUN: Qwen2 is historical; use scripts/qwen_benchmark.py for Qwen3.")
    print()

    # -------------------------------------------------------------------------
    # EVALUATION 1: CodeGPT-small-py on Identifier Splitting
    # -------------------------------------------------------------------------
    print("🧪 EVALUATION 1: microsoft/CodeGPT-small-py on Identifier Splitting")
    print("=" * 70)

    # Load identifier splitting datasets
    identifier_df, identifier_splits = load_identifier_splitting_datasets()

    if len(identifier_df) == 0:
        print("❌ No identifier data loaded. Skipping evaluation 1.")
    else:
        # Initialize CodeGPT segmenter
        print("\n📦 Initializing CodeGPT-small-py segmenter...")

        codegpt_segmenters = {}
        codegpt_segmenters["CodeGPT-small-py"] = HashformersSegmenter(
            segmenter_model="microsoft/CodeGPT-small-py",
            segmenter_type="incremental",
            reranker_model=None,
            reranker_type=None,
        )

        print("✅ CodeGPT segmenter initialized!")

        # Run benchmark
        print("\n🏃 Running CodeGPT evaluation on identifier splitting...")
        codegpt_results = run_benchmark(
            segmenters=codegpt_segmenters,
            dataset=identifier_df,
            input_column="input",
        )

        codegpt_results_df = pd.DataFrame(codegpt_results)
        print("✅ CodeGPT evaluation complete!")

        # Calculate metrics
        codegpt_latency_stats = compute_global_latency(codegpt_results_df)
        codegpt_eval_df = compute_accuracy(codegpt_results_df, identifier_df, codegpt_segmenters)
        codegpt_comparison_df = create_comparison_table(codegpt_results_df)

        # Print results
        print_evaluation_summary(
            codegpt_segmenters, identifier_df, codegpt_latency_stats, codegpt_eval_df,
            "CODEGPT ON IDENTIFIER SPLITTING RESULTS"
        )

        # Export results
        export_results(codegpt_results_df, codegpt_comparison_df, codegpt_latency_stats, "codegpt_identifiers")

    # Clear GPU memory between evaluations
    clear_gpu_memory()

    print("\n" + "=" * 80)
    print("🧪 EVALUATION 2: All Models + RuGPT on NRU HSE Dataset")
    print("=" * 80)

    # Load NRU HSE dataset
    nru_hse_df, nru_hse_split = load_nru_hse_dataset()

    if len(nru_hse_df) == 0:
        print("❌ No NRU HSE data loaded. Skipping evaluation 2.")
        return

    # -------------------------------------------------------------------------
    # EVALUATION 2: All models + RuGPT on NRU HSE
    # -------------------------------------------------------------------------
    print("\n📦 Initializing all segmenters...")

    nru_segmenters = {}

    print("  • WordNinja...")
    nru_segmenters["WordNinja"] = WordNinjaSegmenter()

    try:
        print("  • SymSpell...")
        nru_segmenters["SymSpell"] = SymSpellSegmenter()
    except Exception as e:
        print(f"  ⚠️ SymSpell initialization failed: {e}")

    try:
        print("  • Ekphrasis...")
        nru_segmenters["Ekphrasis"] = EkphrasisSegmenter()
    except Exception as e:
        print(f"  ⚠️ Ekphrasis initialization failed: {e}")

    try:
        print("  • Spiral (Ronin)...")
        nru_segmenters["Spiral-Ronin"] = SpiralSegmenter()
    except Exception as e:
        print(f"  ⚠️ Spiral initialization failed: {e}")

    print("  • Hashformers (GPT-2)...")
    nru_segmenters["Hashformers-GPT2"] = HashformersSegmenter(
        segmenter_model="gpt2",
        segmenter_type="incremental",
        reranker_model=None,
        reranker_type=None,
    )

    print("  • Hashformers (DistilGPT2)...")
    nru_segmenters["Hashformers-DistilGPT2"] = HashformersSegmenter(
        segmenter_model="distilgpt2",
        segmenter_type="incremental",
        reranker_model=None,
        reranker_type=None,
    )

    print("  • Hashformers (RuGPT3Small)...")
    nru_segmenters["Hashformers-RuGPT3Small"] = HashformersSegmenter(
        segmenter_model="ai-forever/rugpt3small_based_on_gpt2",
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
                nru_segmenters["LLM-Qwen2-Historical"] = LocalLLMSegmenter(
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

    print(f"\n✅ Initialized {len(nru_segmenters)} segmenters: {list(nru_segmenters.keys())}")

    # Run benchmark
    print("\n🏃 Running full evaluation on NRU HSE dataset...")
    nru_results = run_benchmark(
        segmenters=nru_segmenters,
        dataset=nru_hse_df,
        input_column="input",
    )

    nru_results_df = pd.DataFrame(nru_results)
    print("✅ NRU HSE evaluation complete!")

    # Check for errors
    errors = nru_results_df[nru_results_df["error"].notna()]
    if len(errors) > 0:
        print(f"  ⚠️ Errors encountered: {len(errors)}")

    # Calculate metrics
    nru_latency_stats = compute_global_latency(nru_results_df)
    nru_eval_df = compute_accuracy(nru_results_df, nru_hse_df, nru_segmenters)
    nru_comparison_df = create_comparison_table(nru_results_df)

    # Print results
    print_evaluation_summary(
        nru_segmenters, nru_hse_df, nru_latency_stats, nru_eval_df,
        "ALL MODELS + RUGPT ON NRU HSE RESULTS"
    )

    # Export results
    export_results(nru_results_df, nru_comparison_df, nru_latency_stats, "all_models_nru_hse")

    print("\n🎉 ALL EVALUATIONS COMPLETE!")
    print("Results saved with suffixes: _codegpt_identifiers, _all_models_nru_hse")


if __name__ == "__main__":
    main()
