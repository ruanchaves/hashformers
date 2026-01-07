#!/usr/bin/env python3
"""
Hashformers Benchmark: Word Segmentation using Hugging Face Hub Datasets

This script benchmarks hashformers against various word segmentation approaches
using datasets loaded from the Hugging Face Hub.

Based on the original benchmark_notebook.ipynb but refactored to:
1. Load datasets from Hugging Face Hub instead of local files
2. Organize datasets into 3 groups: English Hashtags, Foreign Hashtags, Identifier Splitting
3. Calculate latency globally across all groups
4. Calculate accuracy metrics separately per group

Requirements:
    pip install datasets hashformers wordninja symspellpy ekphrasis pandas
"""

import random
import time
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
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


# =============================================================================
# SEGMENTER ARCHITECTURE
# =============================================================================


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


# -----------------------------------------------------------------------------
# WordNinja Segmenter
# -----------------------------------------------------------------------------
import wordninja


class WordNinjaSegmenter(Segmenter):
    """Word segmentation using WordNinja (statistical n-gram model)."""

    def __init__(self):
        """Initialize WordNinja segmenter."""
        pass

    def segment(self, text: str) -> str:
        """Segment text using WordNinja."""
        cleaned = self._clean_input(text)
        words = wordninja.split(cleaned)
        return " ".join(words)


# -----------------------------------------------------------------------------
# SymSpell Segmenter
# -----------------------------------------------------------------------------
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
            import pkg_resources

            try:
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


# -----------------------------------------------------------------------------
# Ekphrasis Segmenter
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Hashformers Segmenter
# -----------------------------------------------------------------------------
from hashformers import TransformerWordSegmenter


class HashformersSegmenter(Segmenter):
    """Word segmentation using Hashformers (Transformer beam search)."""

    def __init__(
        self,
        segmenter_model: str = "gpt2",
        segmenter_type: str = "incremental",
        reranker_model: Optional[str] = None,
        reranker_type: Optional[str] = None,
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
            reranker_model_type=reranker_type,
        )
        self.model_name = segmenter_model

    def segment(self, text: str) -> str:
        """Segment text using Hashformers."""
        cleaned = self._clean_input(text)
        results = self.ws.segment([cleaned])
        return results[0] if results else cleaned


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
        dataset = load_dataset(dataset_name)

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


# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================


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


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    print("=" * 80)
    print("✂️ HASHFORMERS BENCHMARK: Hugging Face Hub Datasets")
    print("=" * 80)
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
