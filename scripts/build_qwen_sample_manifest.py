#!/usr/bin/env python3
"""Build the fixed Qwen benchmark sample manifest from pinned HF datasets.

This utility uses the original sources referenced by the pinned Hugging Face
dataset repositories and only the Python standard library. It deliberately
refuses to regenerate a dataset after its repository revision changes; update
the pin in a reviewed change instead.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("benchmarks/qwen/samples.jsonl")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SAMPLES_PER_DATASET = 20

# These revisions identify the exact dataset repositories used to create the
# committed manifest. HashSet Manual remains excluded, as it was from the 2026
# report, because its hosted dataset build fails.
DATASETS = (
    (
        "English Hashtags",
        "ruanchaves/boun",
        "27f9f67d4662570c17e251438164c3508643c32d",
        "test",
        "hashtag",
    ),
    (
        "English Hashtags",
        "ruanchaves/stan_small",
        "074f7b08b972a1fa3c7ca029a8a99403fac7d048",
        "test",
        "hashtag",
    ),
    (
        "English Hashtags",
        "ruanchaves/stan_large",
        "926842c8fbeadabe99a88d30d4b7ce06a42fb64c",
        "test",
        "hashtag",
    ),
    (
        "English Hashtags",
        "ruanchaves/dev_stanford",
        "292e00146ecc1be6feefdb52362eace417791f4f",
        "validation",
        "hashtag",
    ),
    (
        "English Hashtags",
        "ruanchaves/test_stanford",
        "48f64996c295b22e76cec4454362babfad31f581",
        "test",
        "hashtag",
    ),
    (
        "English Hashtags",
        "ruanchaves/snap",
        "dec0e19ff4bab5b5b1a972909b2ea38118644d0f",
        "train",
        "hashtag",
    ),
    (
        "Foreign Hashtags",
        "ruanchaves/nru_hse",
        "4fb954beab9774a12cac3a13ee08616d5e10df6d",
        "test",
        "hashtag",
    ),
    (
        "Foreign Hashtags",
        "ruanchaves/hashset_distant",
        "0df29003f66c0cb4e17e908cb42e3843d4bd6b11",
        "test",
        "hashtag",
    ),
    (
        "Foreign Hashtags",
        "ruanchaves/hashset_distant_sampled",
        "fb8b329c87153970e0d65e79f8b50220cc2b5ed9",
        "test",
        "hashtag",
    ),
    (
        "Identifier Splitting",
        "ruanchaves/loyola",
        "e51544fd07e72dfa6bf830b56e417adba8dc50ba",
        "test",
        "identifier",
    ),
    (
        "Identifier Splitting",
        "ruanchaves/lynx",
        "9046da8c9a595ead11d7d243780db677f2ce9618",
        "test",
        "identifier",
    ),
    (
        "Identifier Splitting",
        "ruanchaves/jhotdraw",
        "df859ecce54578af17e873cf79438b082632de1d",
        "test",
        "identifier",
    ),
    (
        "Identifier Splitting",
        "ruanchaves/binkley",
        "5ccd62cfd185abd77dffc846d2cd3499e0c286c9",
        "test",
        "identifier",
    ),
    (
        "Identifier Splitting",
        "ruanchaves/bt11",
        "1877395c47bcf77735761c694234dd55d3598bc5",
        "test",
        "identifier",
    ),
)


def request_json(url: str, attempts: int = 6) -> Any:
    """Fetch JSON with bounded retries for transient Hub throttling."""

    request = urllib.request.Request(
        url, headers={"User-Agent": "hashformers-benchmark-manifest/1"}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == attempts - 1:
                raise
        time.sleep(min(2**attempt, 8))
    raise AssertionError("retry loop exhausted")


def request_text(url: str, attempts: int = 6) -> str:
    """Fetch UTF-8 text with bounded transient-error retries."""

    request = urllib.request.Request(
        url, headers={"User-Agent": "hashformers-benchmark-manifest/1"}
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8-sig")
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == attempts - 1:
                raise
        time.sleep(min(2**attempt, 8))
    raise AssertionError("retry loop exhausted")


def dataset_revision(dataset: str) -> str:
    """Return the current repository revision for a hosted dataset."""

    payload = request_json(f"https://huggingface.co/api/datasets/{dataset}")
    return str(payload["sha"])


def selected_indices(
    dataset: str, total: int, count: int = SAMPLES_PER_DATASET
) -> list[int]:
    """Select stable row IDs independently of dataset iteration order."""

    scored = []
    for row_index in range(total):
        key = f"{SEED}\0{dataset}\0{row_index}".encode()
        scored.append((hashlib.sha256(key).digest(), row_index))
    return sorted(row_index for _, row_index in sorted(scored)[: min(total, count)])


def stan_segmentation(hashtag: str, first_gold: str) -> str:
    """Reproduce the hosted STAN dataset's case-preserving alignment."""

    output = ""
    iterator = iter(first_gold.strip())
    for character in hashtag:
        output += character
        while True:
            try:
                next_character = next(iterator)
            except StopIteration:
                break
            if next_character.casefold() == character.casefold():
                break
            if next_character.isspace():
                output = output[:-1] + next_character + output[-1]
    return output


def load_stan(path: Path) -> list[dict[str, str]]:
    """Load STAN CSV records using the same primary-gold conversion as HF."""

    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            golds = ast.literal_eval(row["goldtruths"])
            records.append(
                {
                    "hashtag": row["hashtags"],
                    "segmentation": stan_segmentation(row["hashtags"], golds[0]),
                }
            )
    return records


def align_expansion(identifier: str, expansion: str) -> str:
    """Reproduce Lynx's identifier-to-expanded-text alignment."""

    output = ""
    iterator = iter(expansion)
    for character in identifier:
        while True:
            try:
                next_character = next(iterator)
            except StopIteration:
                break
            if next_character == character:
                output += next_character
                break
            if next_character.isspace():
                output += next_character
    return output


def insert_hyphen_boundaries(segmented: str, identifier: str) -> str:
    """Reproduce the BT11/Binkley hosted conversion from '-' boundaries."""

    output = identifier
    needle = segmented.casefold()
    haystack = identifier.casefold()
    counter = 0
    positions = []
    iterator = iter(haystack)
    for character in needle:
        if character == "-":
            positions.insert(0, counter)
            continue
        while True:
            try:
                next_character = next(iterator)
            except StopIteration:
                break
            counter += 1
            if next_character == character:
                break
    while positions:
        position = positions.pop(0)
        output = output[:position] + " " + output[position:]
    return output


def insert_loyola_boundaries(segmented: str, identifier: str) -> str:
    """Reproduce Loyola's hyphen and punctuation boundary conversion."""

    output = insert_hyphen_boundaries(segmented, identifier)
    positions = []
    previous = output[0]
    for index, character in enumerate(output[1:]):
        if not previous.isalnum() and not previous.isspace() and character.isalnum():
            positions.insert(0, index + 1)
        previous = character
    while positions:
        position = positions.pop(0)
        output = output[:position] + " " + output[position:]
    return output


def load_segmented_lines(text: str) -> list[dict[str, str]]:
    """Convert one already-segmented hashtag per line to hosted records."""

    records = []
    for line in text.splitlines():
        segmentation = line.strip()
        if segmentation:
            records.append(
                {
                    "hashtag": segmentation.replace(" ", ""),
                    "segmentation": segmentation,
                }
            )
    return records


def load_test_stanford() -> list[dict[str, str]]:
    """Reproduce the hosted candidate-ranking dataset's gold extraction."""

    path = REPOSITORY_ROOT / "datasets/Test-Stanford.txt"
    grouped = []
    current_hashtag = None
    candidates: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle, delimiter="\t")
        next(rows, None)
        for row in rows:
            if len(row) < 4:
                continue
            hashtag = row[1].strip("'").strip()
            candidate = row[2].strip("'").strip()
            label = int(row[3])
            if current_hashtag != hashtag:
                if current_hashtag is not None:
                    grouped.append((current_hashtag, candidates, labels))
                current_hashtag = hashtag
                candidates = [candidate]
                labels = [label]
            else:
                candidates.append(candidate)
                labels.append(label)
    # The hosted generator omitted its final accumulation; retain parity with
    # the pinned 1,261-row HF split rather than silently changing the sample set.
    records = []
    for hashtag, candidates, labels in grouped:
        try:
            segmentation = candidates[labels.index(1)]
        except ValueError:
            continue
        records.append({"hashtag": hashtag, "segmentation": segmentation})
    return records


def load_hashset(url: str) -> list[dict[str, str]]:
    """Load the original HashSet CSV schema referenced by its HF builder."""

    records = []
    for row in csv.DictReader(io.StringIO(request_text(url))):
        records.append(
            {
                "hashtag": row["Unsegmented_hashtag"],
                "segmentation": row["Segmented_hashtag"],
            }
        )
    return records


def load_local_dataset(dataset: str) -> list[dict[str, str]]:
    """Load datasets whose legacy Hub scripts no longer have a server preview."""

    if dataset == "ruanchaves/boun":
        return load_segmented_lines(
            request_text(
                "https://raw.githubusercontent.com/ardax/hashtag-segmentor/master/Test-BOUN"
            )
        )
    if dataset == "ruanchaves/stan_small":
        return load_stan(REPOSITORY_ROOT / "datasets/stan_small.csv")
    if dataset == "ruanchaves/stan_large":
        return load_stan(REPOSITORY_ROOT / "datasets/stan_large_test.csv")
    if dataset == "ruanchaves/dev_stanford":
        return load_segmented_lines(
            request_text(
                "https://raw.githubusercontent.com/ardax/hashtag-segmentor/master/Dev-Stanford"
            )
        )
    if dataset == "ruanchaves/test_stanford":
        return load_test_stanford()
    if dataset == "ruanchaves/snap":
        return load_segmented_lines(
            (
                REPOSITORY_ROOT / "datasets/SNAP.Hashtags.Segmented.w.Heuristics.txt"
            ).read_text(encoding="utf-8")
        )
    if dataset == "ruanchaves/nru_hse":
        text = request_text(
            "https://raw.githubusercontent.com/glushkovato/hashtag_segmentation/"
            "master/data/test_rus.csv"
        )
        records = []
        for row in csv.DictReader(io.StringIO(text)):
            hashtag = row["hashtag"]
            labels = row["true_segmentation"]
            segmentation = "".join(
                character + (" " if label == "1" else "")
                for character, label in zip(hashtag, labels)
            ).strip()
            records.append({"hashtag": hashtag, "segmentation": segmentation})
        return records
    if dataset == "ruanchaves/hashset_distant":
        return load_hashset(
            "https://raw.githubusercontent.com/prashantkodali/HashSet/"
            "master/datasets/hashset/HashSet-Distant.csv"
        )
    if dataset == "ruanchaves/hashset_distant_sampled":
        return load_hashset(
            "https://raw.githubusercontent.com/prashantkodali/HashSet/"
            "master/datasets/hashset/HashSet-Distant-sampled.csv"
        )
    if dataset == "ruanchaves/loyola":
        records = []
        with (
            REPOSITORY_ROOT
            / "datasets/loyola-udelaware-identifier-splitting-oracle.txt"
        ).open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split(" ")
                if len(fields) >= 5:
                    records.append(
                        {
                            "identifier": fields[1],
                            "segmentation": insert_loyola_boundaries(
                                fields[4], fields[1]
                            ),
                        }
                    )
        return records
    if dataset in {"ruanchaves/lynx", "ruanchaves/jhotdraw"}:
        name = dataset.rsplit("/", 1)[1]
        records = []
        with (REPOSITORY_ROOT / f"datasets/{name}.txt").open(
            "r", encoding="utf-8"
        ) as handle:
            for line in handle:
                identifier, annotation = line.split(":", 1)
                identifier = identifier.strip()
                annotation = annotation.strip()
                segmentation = (
                    align_expansion(identifier, annotation)
                    if name == "lynx"
                    else annotation
                )
                records.append({"identifier": identifier, "segmentation": segmentation})
        return records
    if dataset in {"ruanchaves/binkley", "ruanchaves/bt11"}:
        name = dataset.rsplit("/", 1)[1]
        records = []
        with (REPOSITORY_ROOT / f"datasets/{name}.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.reader(handle):
                if len(row) >= 2 and row[0] and row[1]:
                    records.append(
                        {
                            "identifier": row[0],
                            "segmentation": insert_hyphen_boundaries(row[1], row[0]),
                        }
                    )
        return records
    raise KeyError(f"no local loader for {dataset}")


def build_records() -> list[dict[str, Any]]:
    """Fetch and validate every pinned dataset record."""

    records = []
    for group, dataset, expected_revision, split, input_field in DATASETS:
        before_revision = dataset_revision(dataset)
        if before_revision != expected_revision:
            raise RuntimeError(
                f"{dataset}: expected revision {expected_revision}, found {before_revision}; "
                "review the dataset change before updating the pin"
            )
        all_rows = load_local_dataset(dataset)
        indices = selected_indices(dataset, len(all_rows))
        rows = {row_index: all_rows[row_index] for row_index in indices}
        after_revision = dataset_revision(dataset)
        if after_revision != before_revision:
            raise RuntimeError(
                f"{dataset}: repository revision changed while building the manifest"
            )
        for row_index in indices:
            row = rows[row_index]
            if input_field not in row or "segmentation" not in row:
                raise RuntimeError(
                    f"{dataset}/{split}:{row_index}: expected {input_field!r} and 'segmentation' fields"
                )
            source = str(row[input_field]).lstrip("#").strip()
            gold = str(row["segmentation"]).strip()
            if not source or not gold:
                raise RuntimeError(
                    f"{dataset}/{split}:{row_index}: empty input or gold segmentation"
                )
            records.append(
                {
                    "sample_id": f"{dataset}@{split}:{row_index}",
                    "dataset": dataset,
                    "dataset_revision": expected_revision,
                    "split": split,
                    "row_index": row_index,
                    "group": group,
                    "input": source,
                    "gold": gold,
                }
            )
    return records


def write_jsonl(
    path: Path, records: Sequence[Mapping[str, Any]], overwrite: bool
) -> None:
    """Atomically write the manifest, protecting a prior fixed sample set."""

    if path.exists() and not overwrite:
        raise SystemExit(
            f"refusing to replace existing manifest: {path}; pass --overwrite explicitly"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = build_records()
    write_jsonl(args.output, records, args.overwrite)
    print(f"wrote {len(records)} fixed samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
