"""Expose Hashformers segmentation workflows through a local MCP server.

"""

import csv
import hashlib
import json
import math
import os
import secrets
import sqlite3
import stat
from argparse import ArgumentParser
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, TextIO

import anyio
import regex as timeout_regex
import torch
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.types import ToolAnnotations
from typing_extensions import TypedDict

from hashformers.beamsearch.data_structures import ProbabilityDictionary
from hashformers.segmenter import (
    BaseWordSegmenter,
    RegexWordSegmenter,
    TweetSegmenter,
    TwitterTextMatcher,
)
from hashformers.segmenter.auto import TransformerWordSegmenter
from hashformers.segmenter.data_structures import HashtagContainer, WordSegmenterOutput

RankingStrategy = Literal["auto", "segmenter", "reranker", "ensemble"]
ResolvedRankingStrategy = Literal["segmenter", "reranker", "ensemble"]
SegmenterKind = Literal["regex", "transformer"]
FileInputFormat = Literal["auto", "text", "csv", "jsonl"]

MAX_INTERACTIVE_INPUTS = 64
MAX_FILE_UNIQUE_HASHTAGS = 64
MAX_TOP_K = 64
MAX_STEPS = 32
MAX_BEAM_EXPANSIONS = 250_000
MAX_RETURNED_CANDIDATES = 64
MAX_PRECOMPUTED_CANDIDATES = 4096
MAX_HASHTAG_LENGTH = 512
MAX_REGEX_RULES = 32
MAX_REGEX_PATTERN_LENGTH = 512
MAX_REGEX_INPUT_LENGTH = 10_000
MAX_REGEX_OUTPUT_LENGTH = 20_000
MAX_TWEET_HASHTAG_OCCURRENCES = 64
MAX_TWEET_REPLACEMENT_CHARS = 128
MAX_TWEET_OUTPUT_CHARS = 1_000_000
MAX_FILE_RECORD_CHARS = 65_536
MAX_JOB_METADATA_CHARS = 65_536
MAX_JOB_RESULT_CHARS = 1_000_000
REGEX_TIMEOUT_SECONDS = 0.1
JOB_APPLICATION_ID = 0x48464D52
JOB_SCHEMA_VERSION = 1
JOB_METADATA_KEYS = (
    "input_path",
    "output_path",
    "input_format",
    "input_field",
    "input_hash",
    "overwrite",
    "finalized",
    "server_config",
    "run_options",
)
JOB_TABLE_SCHEMAS = {
    "metadata": (
        ("key", "TEXT", 0, 1),
        ("value", "TEXT", 1, 0),
    ),
    "source_records": (
        ("source_line", "INTEGER", 0, 1),
        ("input", "TEXT", 1, 0),
        ("normalized", "TEXT", 1, 0),
    ),
    "unique_hashtags": (
        ("normalized", "TEXT", 0, 1),
        ("representative_input", "TEXT", 1, 0),
        ("result_json", "TEXT", 0, 0),
    ),
}
SUPPORTS_SECURE_DIR_FD = (
    os.name != "nt"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.open in os.supports_dir_fd
    and os.link in os.supports_dir_fd
    and os.link in os.supports_follow_symlinks
    and os.rename in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
)
DESCRIPTOR_DIRECTORY = (
    Path("/proc/self/fd") if Path("/proc/self/fd").is_dir() else None
)
SUPPORTS_SECURE_FILE_JOBS = (
    SUPPORTS_SECURE_DIR_FD and DESCRIPTOR_DIRECTORY is not None
)
SECURE_FILE_JOB_ERROR = (
    "secure file jobs are not supported on this platform; use a Linux host "
    "with /proc/self/fd"
)


class Candidate(TypedDict):
    """Represent one ranked segmentation candidate.

    Attributes:
        segmentation: Candidate text with inferred word boundaries.
        score: Score used for ordering; lower values rank higher.
        rank: One-based position within the candidate set.
    """

    segmentation: str
    score: float
    rank: int


class ComponentRankings(TypedDict):
    """Represent rankings produced by each configured pipeline component.

    Attributes:
        segmenter: Candidates scored by the beam-search model.
        reranker: Candidates scored by the reranker, when used.
        ensemble: Candidates selected by the ensemble, when used.
    """

    segmenter: list[Candidate]
    reranker: list[Candidate] | None
    ensemble: list[Candidate] | None


class SegmentationResult(TypedDict):
    """Represent a complete segmentation result for one input.

    Attributes:
        input: Original text supplied by the caller.
        normalized_input: Text after Hashformers preprocessing.
        selected_segmentation: Highest-ranked segmentation.
        ranking_strategy: Component that selected the result.
        candidates: Ranked candidates from the selected component.
        component_rankings: Optional rankings for every component that ran.
    """

    input: str
    normalized_input: str
    selected_segmentation: str
    ranking_strategy: ResolvedRankingStrategy
    candidates: list[Candidate]
    component_rankings: ComponentRankings | None


class SegmentationsResult(TypedDict):
    """Represent a list of segmentation results.

    Attributes:
        results: Per-input results in input order.
    """

    results: list[SegmentationResult]


class RegexSegmentationResult(TypedDict):
    """Represent one deterministic regex segmentation.

    Attributes:
        input: Original text supplied by the caller.
        normalized_input: Text after Hashformers preprocessing.
        segmentation: Text after applying every regex rule in order.
    """

    input: str
    normalized_input: str
    segmentation: str


class RegexSegmentationsResult(TypedDict):
    """Represent regex segmentation results.

    Attributes:
        results: Per-input results in input order.
    """

    results: list[RegexSegmentationResult]


class TweetSegmentationResult(TypedDict):
    """Represent one tweet after its hashtags have been segmented.

    Attributes:
        input: Original tweet.
        segmented_text: Tweet with segmented hashtags substituted.
        hashtags: Segmentation details in hashtag occurrence order.
    """

    input: str
    segmented_text: str
    hashtags: list[SegmentationResult]


class TweetSegmentationsResult(TypedDict):
    """Represent tweet segmentation results.

    Attributes:
        results: Per-tweet results in input order.
    """

    results: list[TweetSegmentationResult]


class ScoredCandidateInput(TypedDict):
    """Represent a caller-supplied candidate and segmenter score.

    Attributes:
        segmentation: Candidate segmentation.
        score: Existing lower-is-better segmenter score.
    """

    segmentation: str
    score: float


class CandidateSetInput(TypedDict):
    """Represent precomputed candidates for one unsegmented input.

    Attributes:
        input: Original unsegmented text.
        candidates: Hypotheses with their existing segmenter scores.
    """

    input: str
    candidates: list[ScoredCandidateInput]


class FileJobStatus(TypedDict):
    """Summarize a resumable file job without returning segmentation data.

    Attributes:
        job_path: Persistent local checkpoint passed to continuation calls.
        input_path: Absolute path read by the MCP server.
        output_path: Absolute JSON Lines path written by the MCP server.
        input_format: Resolved input format.
        status: Whether more continuation calls are required.
        total_hashtags: Number of source records indexed.
        unique_hashtags: Number of distinct normalized inputs encountered.
        deduplicated_hashtags: Number of repeated inputs reused from cache.
        processed_unique: Number of unique hashtags checkpointed.
        processed_this_call: Number checkpointed by the latest continuation.
        remaining_unique: Number still requiring inference.
        segmenter_model: Model that generated beam-search candidates.
        reranker_model: Optional model used for reranking.
        ranking_strategy: Component used to select output segmentations.
    """

    job_path: str
    input_path: str
    output_path: str
    input_format: str
    status: Literal["in_progress", "completed"]
    total_hashtags: int
    unique_hashtags: int
    deduplicated_hashtags: int
    processed_unique: int
    processed_this_call: int
    remaining_unique: int
    segmenter_model: str
    reranker_model: str | None
    ranking_strategy: ResolvedRankingStrategy


@dataclass(frozen=True)
class ServerConfig:
    """Configure models and memory policy for one MCP server process.

    Attributes:
        segmenter_model: Hugging Face model name or local path.
        segmenter_model_type: Hashformers scorer type for beam search.
        segmenter_device: Device string or ``auto``.
        segmenter_batch_size: Maximum segmenter inference batch size.
        reranker_model: Optional Hugging Face reranker name or local path.
        reranker_model_type: Hashformers scorer type for reranking.
        reranker_device: Device string or ``inherit``.
        reranker_batch_size: Maximum reranker inference batch size.
        file_roots: Directories available to file-job tools.
        allow_file_overwrite: Whether callers may replace existing files.
    """

    segmenter_model: str = "gpt2"
    segmenter_model_type: str = "gpt2"
    segmenter_device: str = "auto"
    segmenter_batch_size: int = 64
    reranker_model: str | None = None
    reranker_model_type: str = "bert"
    reranker_device: str = "inherit"
    reranker_batch_size: int = 64
    file_roots: tuple[str, ...] = (".",)
    allow_file_overwrite: bool = False


@dataclass(frozen=True)
class _ConfiguredFileRoot:
    """Pin one configured file root to its startup filesystem identity.

    Attributes:
        path: Canonical path captured when the server was configured.
        descriptor: Process-owned descriptor for the configured directory.
        device: Filesystem device number for the configured directory.
        inode: Filesystem inode number for the configured directory.
    """

    path: Path
    descriptor: int
    device: int
    inode: int


_server_config = ServerConfig()
_configured_file_roots: tuple[_ConfiguredFileRoot, ...] = ()
_segmenter: TransformerWordSegmenter | None = None
_segmenter_lock = Lock()
_file_roots_lock = Lock()
_inference_lock = Lock()


def _positive_integer(value: str) -> int:
    """Parse a positive integer for the command-line interface.

    Args:
        value: Raw command-line value.

    Returns:
        The parsed positive integer.

    Raises:
        ValueError: If the value is not a positive integer.
    """
    parsed = int(value)
    if parsed < 1:
        raise ValueError("must be a positive integer")
    return parsed


def build_argument_parser() -> ArgumentParser:
    """Build the MCP server command-line parser.

    Returns:
        Parser for model, device, and batch-size configuration.
    """
    parser = ArgumentParser(
        prog="hashformers-mcp",
        description="Run the Hashformers MCP server over stdio.",
    )
    parser.add_argument(
        "--model",
        "--segmenter-model",
        dest="segmenter_model",
        default="gpt2",
        help="Hugging Face model name or local path used for beam search.",
    )
    parser.add_argument(
        "--segmenter-model-type",
        default="gpt2",
        help="Hashformers scorer type used for beam search.",
    )
    parser.add_argument(
        "--device",
        "--segmenter-device",
        dest="segmenter_device",
        default="auto",
        help="Segmenter device string; 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--batch-size",
        "--segmenter-batch-size",
        dest="segmenter_batch_size",
        type=_positive_integer,
        default=64,
        help="Maximum segmenter inference batch size.",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Optional Hugging Face reranker name or local path.",
    )
    parser.add_argument(
        "--reranker-model-type",
        default="bert",
        help="Hashformers scorer type used for reranking.",
    )
    parser.add_argument(
        "--reranker-device",
        default="inherit",
        help="Reranker device string; 'inherit' uses the segmenter device.",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=_positive_integer,
        default=64,
        help="Maximum reranker inference batch size.",
    )
    parser.add_argument(
        "--file-root",
        action="append",
        default=None,
        help=(
            "Directory file-job tools may read and write. Repeat for multiple "
            "roots; defaults to the server working directory."
        ),
    )
    parser.add_argument(
        "--allow-file-overwrite",
        action="store_true",
        help="Allow file jobs with overwrite=true to replace existing output.",
    )
    return parser


def parse_server_config(argv: list[str] | None = None) -> ServerConfig:
    """Parse process-wide MCP server configuration.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Validated server configuration.
    """
    arguments = vars(build_argument_parser().parse_args(argv))
    arguments["file_roots"] = tuple(arguments.pop("file_root") or ["."])
    config = ServerConfig(**arguments)
    _validate_server_config(config)
    return config


def _validate_server_config(config: ServerConfig) -> None:
    """Validate model and memory settings.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If a required string is blank or a batch size is invalid.
    """
    required_strings = {
        "segmenter_model": config.segmenter_model,
        "segmenter_model_type": config.segmenter_model_type,
        "segmenter_device": config.segmenter_device,
        "reranker_model_type": config.reranker_model_type,
        "reranker_device": config.reranker_device,
    }
    if config.reranker_model is not None:
        required_strings["reranker_model"] = config.reranker_model
    for name, value in required_strings.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain text")
    for name, value in {
        "segmenter_batch_size": config.segmenter_batch_size,
        "reranker_batch_size": config.reranker_batch_size,
    }.items():
        _validate_positive(value, name)
    if not isinstance(config.file_roots, tuple) or not config.file_roots:
        raise ValueError("file_roots must contain at least one directory")
    for root in config.file_roots:
        if not isinstance(root, str) or not root.strip():
            raise ValueError("file_roots must contain directory paths")
        if not Path(root).expanduser().resolve().is_dir():
            raise ValueError(f"file root is not a directory: {root}")
    if not isinstance(config.allow_file_overwrite, bool):
        raise ValueError("allow_file_overwrite must be a boolean")


def configure_server(config: ServerConfig) -> None:
    """Set process-wide configuration before serving requests.

    Args:
        config: Model and memory settings for the server process.

    Returns:
        None.
    """
    global _configured_file_roots, _server_config, _segmenter
    _validate_server_config(config)
    configured_file_roots = _pin_file_roots(config.file_roots)
    with _segmenter_lock, _file_roots_lock:
        previous_file_roots = _configured_file_roots
        _server_config = config
        _configured_file_roots = configured_file_roots
        _segmenter = None
    for root in previous_file_roots:
        if root.descriptor >= 0:
            os.close(root.descriptor)


def _model_config_payload() -> dict[str, object]:
    """Return only the configuration that affects model output.

    Returns:
        JSON-safe model, device, and batch-size configuration.
    """
    return {
        "segmenter_model": _server_config.segmenter_model,
        "segmenter_model_type": _server_config.segmenter_model_type,
        "segmenter_device": _server_config.segmenter_device,
        "segmenter_batch_size": _server_config.segmenter_batch_size,
        "reranker_model": _server_config.reranker_model,
        "reranker_model_type": _server_config.reranker_model_type,
        "reranker_device": _server_config.reranker_device,
        "reranker_batch_size": _server_config.reranker_batch_size,
    }


def _resolve_file_path(
    raw_path: str | Path,
    name: str,
    *,
    must_exist: bool,
) -> Path:
    """Resolve and authorize a path against process-wide file roots.

    Args:
        raw_path: Caller- or checkpoint-provided path.
        name: Argument name used in an error message.
        must_exist: Whether every path component must already exist.

    Returns:
        Canonical authorized path.

    Raises:
        ValueError: If the path is missing or outside configured roots.
    """
    try:
        resolved = Path(raw_path).expanduser().resolve(strict=must_exist)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{name} does not exist: {raw_path}") from error
    roots = _active_file_roots()
    if not any(
        resolved == root.path or resolved.is_relative_to(root.path)
        for root in roots
    ):
        allowed = ", ".join(str(root.path) for root in roots)
        raise ValueError(f"{name} is outside configured file roots ({allowed})")
    return resolved


def _configured_root_for(path: Path) -> _ConfiguredFileRoot:
    """Return the canonical configured root containing a resolved path.

    Args:
        path: Canonical path already authorized by ``_resolve_file_path``.

    Returns:
        The narrowest pinned root containing the path.

    Raises:
        ValueError: If no active file root contains the path.
    """
    roots = sorted(
        _active_file_roots(),
        key=lambda root: len(root.path.parts),
        reverse=True,
    )
    for root in roots:
        if path == root.path or path.is_relative_to(root.path):
            return root
    raise ValueError("path is outside configured file roots")


def _open_directory_nofollow(path: Path) -> int:
    """Open an absolute directory without following path-component symlinks.

    Args:
        path: Canonical absolute directory path.

    Returns:
        An owned directory file descriptor.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    if os.name == "nt" or not hasattr(os, "O_NOFOLLOW"):
        return os.open(path, flags)
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            child = os.open(
                part,
                flags | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _pin_file_roots(
    raw_roots: tuple[str, ...],
) -> tuple[_ConfiguredFileRoot, ...]:
    """Capture canonical paths and filesystem identities for file roots.

    Args:
        raw_roots: Operator-configured root directory paths.

    Returns:
        Immutable root records used for every later authorization check.

    Raises:
        ValueError: If a configured root cannot be opened as a directory.
    """
    configured: list[_ConfiguredFileRoot] = []
    try:
        for raw_root in raw_roots:
            root = Path(raw_root).expanduser().resolve(strict=True)
            if not SUPPORTS_SECURE_FILE_JOBS:
                root_stat = root.stat()
                configured.append(
                    _ConfiguredFileRoot(
                        path=root,
                        descriptor=-1,
                        device=root_stat.st_dev,
                        inode=root_stat.st_ino,
                    )
                )
                continue
            descriptor = _open_directory_nofollow(root)
            try:
                root_stat = os.fstat(descriptor)
                if not stat.S_ISDIR(root_stat.st_mode):
                    raise ValueError(
                        f"file root is not a directory: {raw_root}"
                    )
            except BaseException:
                os.close(descriptor)
                raise
            configured.append(
                _ConfiguredFileRoot(
                    path=root,
                    descriptor=descriptor,
                    device=root_stat.st_dev,
                    inode=root_stat.st_ino,
                )
            )
    except BaseException:
        for configured_root in configured:
            if configured_root.descriptor >= 0:
                os.close(configured_root.descriptor)
        raise
    return tuple(configured)


def _active_file_roots() -> tuple[_ConfiguredFileRoot, ...]:
    """Return roots pinned by explicit or lazy server configuration.

    Returns:
        Immutable configured root records.
    """
    global _configured_file_roots
    if _configured_file_roots:
        return _configured_file_roots
    with _file_roots_lock:
        if not _configured_file_roots:
            _configured_file_roots = _pin_file_roots(_server_config.file_roots)
        return _configured_file_roots


def _open_pinned_root(root: _ConfiguredFileRoot) -> int:
    """Open a configured root only if its startup identity is unchanged.

    Args:
        root: Root path and identity captured during configuration.

    Returns:
        An owned descriptor for the same configured directory.

    Raises:
        ValueError: If the configured path was replaced after startup.
    """
    descriptor = None
    visible_descriptor = None
    try:
        with _file_roots_lock:
            if root not in _configured_file_roots:
                raise ValueError("server file-root configuration changed")
            descriptor = os.dup(root.descriptor)
        visible_descriptor = _open_directory_nofollow(root.path)
        visible_stat = os.fstat(visible_descriptor)
        if (visible_stat.st_dev, visible_stat.st_ino) != (
            root.device,
            root.inode,
        ):
            raise ValueError(
                f"configured file root changed after server startup: {root.path}"
            )
        os.close(visible_descriptor)
        visible_descriptor = None
        return descriptor
    except OSError as error:
        if visible_descriptor is not None:
            os.close(visible_descriptor)
        if descriptor is not None:
            os.close(descriptor)
        raise ValueError(
            f"configured file root changed after server startup: {root.path}"
        ) from error
    except BaseException:
        if visible_descriptor is not None:
            os.close(visible_descriptor)
        if descriptor is not None:
            os.close(descriptor)
        raise


def _descriptor_path(descriptor: int) -> Path:
    """Build a path that remains bound to an already-open descriptor.

    Args:
        descriptor: Open file or directory descriptor.

    Returns:
        Descriptor-backed path suitable for SQLite.

    Raises:
        ValueError: If the platform lacks descriptor-backed filesystem paths.
    """
    if not SUPPORTS_SECURE_FILE_JOBS or DESCRIPTOR_DIRECTORY is None:
        raise ValueError(SECURE_FILE_JOB_ERROR)
    path = DESCRIPTOR_DIRECTORY / str(descriptor)
    try:
        if not os.path.samestat(os.fstat(descriptor), path.stat()):
            raise ValueError(SECURE_FILE_JOB_ERROR)
    except OSError as error:
        raise ValueError(SECURE_FILE_JOB_ERROR) from error
    return path


def _connect_descriptor_database(
    descriptor: int,
    *,
    timeout: float = 0,
) -> sqlite3.Connection:
    """Open SQLite through a verified descriptor without pathname fallback.

    Args:
        descriptor: Open descriptor for an existing regular database file.
        timeout: SQLite lock wait in seconds.

    Returns:
        A connection to the same file inode as ``descriptor``.

    Raises:
        ValueError: If SQLite cannot preserve the descriptor-bound identity.
    """
    descriptor_path = _descriptor_path(descriptor)
    try:
        connection = sqlite3.connect(
            f"file:{descriptor_path}?mode=rw",
            uri=True,
            timeout=timeout,
        )
    except sqlite3.OperationalError as error:
        raise ValueError(SECURE_FILE_JOB_ERROR) from error
    try:
        database_rows = connection.execute("PRAGMA database_list").fetchall()
        main_rows = [row for row in database_rows if row[1] == "main"]
        if len(main_rows) != 1 or not main_rows[0][2]:
            raise ValueError(SECURE_FILE_JOB_ERROR)
        database_path = Path(main_rows[0][2])
        if (
            DESCRIPTOR_DIRECTORY is not None
            and (
                database_path == DESCRIPTOR_DIRECTORY
                or database_path.is_relative_to(DESCRIPTOR_DIRECTORY)
            )
        ):
            raise ValueError(SECURE_FILE_JOB_ERROR)
        try:
            same_file = os.path.samestat(
                os.fstat(descriptor),
                database_path.stat(),
            )
        except OSError as error:
            raise ValueError(SECURE_FILE_JOB_ERROR) from error
        if not same_file:
            raise ValueError(SECURE_FILE_JOB_ERROR)
        return connection
    except BaseException:
        connection.close()
        raise


def _open_authorized_parent(
    raw_path: str | Path,
    name: str,
    *,
    must_exist: bool,
) -> tuple[Path, int | None]:
    """Resolve a path and anchor its parent to a no-follow descriptor.

    Args:
        raw_path: Caller- or checkpoint-provided path.
        name: Argument name used in errors.
        must_exist: Whether the final path must already exist.

    Returns:
        The canonical path and an owned descriptor for its parent directory.

    Raises:
        ValueError: If the path is a configured root rather than a child path.
    """
    if not SUPPORTS_SECURE_FILE_JOBS:
        raise ValueError(SECURE_FILE_JOB_ERROR)
    resolved = _resolve_file_path(raw_path, name, must_exist=must_exist)
    root = _configured_root_for(resolved)
    if resolved == root.path:
        raise ValueError(f"{name} must identify a file inside a configured root")
    descriptor = _open_pinned_root(root)
    try:
        for part in resolved.parent.relative_to(root.path).parts:
            child = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return resolved, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_child(
    parent_descriptor: int | None,
    parent_path: Path,
    name: str,
    flags: int,
    error_message: str,
) -> int:
    """Open an untrusted child without blocking and require a regular file.

    Args:
        parent_descriptor: Anchored directory descriptor or ``None``.
        parent_path: Canonical fallback parent path.
        name: Child basename.
        flags: Access flags accepted by ``os.open``.
        error_message: Validation error raised for a special file.

    Returns:
        An owned descriptor for a regular file.

    Raises:
        ValueError: If the opened child is not a regular file.
    """
    descriptor = _open_child(
        parent_descriptor,
        parent_path,
        name,
        flags
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(error_message)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_child(
    parent_descriptor: int | None,
    parent_path: Path,
    name: str,
    flags: int,
    mode: int = 0o777,
) -> int:
    """Open a child through its anchored parent when supported.

    Args:
        parent_descriptor: Anchored directory descriptor or ``None``.
        parent_path: Canonical fallback parent path.
        name: Child basename.
        flags: Flags accepted by ``os.open``.
        mode: Creation permissions when relevant.

    Returns:
        An owned file descriptor.
    """
    if parent_descriptor is None:
        return os.open(parent_path / name, flags, mode)
    return os.open(name, flags, mode, dir_fd=parent_descriptor)


def _unlink_child(
    parent_descriptor: int | None,
    parent_path: Path,
    name: str,
) -> None:
    """Unlink a child through an anchored parent when supported.

    Args:
        parent_descriptor: Anchored directory descriptor or ``None``.
        parent_path: Canonical fallback parent path.
        name: Child basename.

    Returns:
        None.
    """
    if parent_descriptor is None:
        os.unlink(parent_path / name)
    else:
        os.unlink(name, dir_fd=parent_descriptor)


def _link_children(
    parent_descriptor: int | None,
    parent_path: Path,
    source: str,
    destination: str,
) -> None:
    """Publish a child with atomic no-replace semantics.

    Args:
        parent_descriptor: Anchored directory descriptor or ``None``.
        parent_path: Canonical fallback parent path.
        source: Existing temporary basename.
        destination: New destination basename.

    Returns:
        None.
    """
    if parent_descriptor is None:
        os.link(parent_path / source, parent_path / destination)
    else:
        os.link(
            source,
            destination,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )


def _replace_child(
    parent_descriptor: int | None,
    parent_path: Path,
    source: str,
    destination: str,
) -> None:
    """Atomically replace one sibling with another.

    Args:
        parent_descriptor: Anchored directory descriptor or ``None``.
        parent_path: Canonical fallback parent path.
        source: Existing temporary basename.
        destination: Destination basename.

    Returns:
        None.
    """
    if parent_descriptor is None:
        os.replace(parent_path / source, parent_path / destination)
    else:
        os.replace(
            source,
            destination,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )


def _fsync_parent(
    parent_descriptor: int | None,
    parent_path: Path,
) -> None:
    """Persist directory entries when the platform supports it.

    Args:
        parent_descriptor: Anchored directory descriptor or ``None``.
        parent_path: Canonical fallback directory path.

    Returns:
        None.
    """
    if parent_descriptor is not None:
        os.fsync(parent_descriptor)
        return
    if os.name == "nt":
        return
    descriptor = os.open(
        parent_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _connect_checkpoint(
    checkpoint: Path,
    *,
    timeout: float = 0,
):
    """Open SQLite against a no-follow descriptor for the checkpoint inode.

    Args:
        checkpoint: Authorized checkpoint path.
        timeout: SQLite lock wait in seconds.

    Yields:
        An open SQLite connection anchored to the verified checkpoint file.

    Raises:
        ValueError: If the checkpoint is not a regular file.
    """
    resolved, parent_descriptor = _open_authorized_parent(
        checkpoint,
        "job_path",
        must_exist=True,
    )
    file_descriptor = None
    connection = None
    try:
        file_descriptor = _open_regular_child(
            parent_descriptor,
            resolved.parent,
            resolved.name,
            os.O_RDWR,
            f"job_path is not a readable checkpoint: {resolved}",
        )
        connection = _connect_descriptor_database(
            file_descriptor,
            timeout=timeout,
        )
        yield connection
    finally:
        if connection is not None:
            connection.close()
        if file_descriptor is not None:
            os.close(file_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _resolve_device(device: str, inherited_device: str | None = None) -> str:
    """Resolve MCP convenience device values.

    Args:
        device: Configured device string.
        inherited_device: Resolved segmenter device for ``inherit``.

    Returns:
        Concrete device string accepted by Hashformers.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "inherit":
        if inherited_device is None:
            raise ValueError("inherit requires a resolved segmenter device")
        return inherited_device
    return device


def get_segmenter() -> TransformerWordSegmenter:
    """Return the lazily initialized process-wide Transformer segmenter.

    Returns:
        The segmenter reused by every Transformer-backed MCP tool call.
    """
    global _segmenter
    if _segmenter is None:
        with _segmenter_lock:
            if _segmenter is None:
                segmenter_device = _resolve_device(_server_config.segmenter_device)
                reranker_device = _resolve_device(
                    _server_config.reranker_device,
                    inherited_device=segmenter_device,
                )
                _segmenter = TransformerWordSegmenter(
                    segmenter_model_name_or_path=_server_config.segmenter_model,
                    segmenter_model_type=_server_config.segmenter_model_type,
                    segmenter_device=segmenter_device,
                    segmenter_gpu_batch_size=_server_config.segmenter_batch_size,
                    reranker_model_name_or_path=_server_config.reranker_model,
                    reranker_model_type=_server_config.reranker_model_type,
                    reranker_device=reranker_device,
                    reranker_gpu_batch_size=_server_config.reranker_batch_size,
                )
    return _segmenter


def _validate_positive(value: int, name: str) -> None:
    """Validate a positive non-boolean integer.

    Args:
        value: Value to validate.
        name: Argument name used in an error message.

    Raises:
        ValueError: If the value is not a positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_at_most(value: int, maximum: int, name: str) -> None:
    """Validate a positive integer with a server-enforced upper bound.

    Args:
        value: Value to validate.
        maximum: Largest accepted value.
        name: Argument name used in an error message.

    Raises:
        ValueError: If the value is not positive or exceeds ``maximum``.
    """
    _validate_positive(value, name)
    if value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def _validate_input_count(values: list, name: str) -> None:
    """Keep interactive MCP requests within a fixed memory budget.

    Args:
        values: Caller-provided list.
        name: Argument name used in an error message.

    Raises:
        ValueError: If the list exceeds the interactive request ceiling.
    """
    if len(values) > MAX_INTERACTIVE_INPUTS:
        raise ValueError(
            f"{name} accepts at most {MAX_INTERACTIVE_INPUTS} items; "
            "use the resumable file-job tools for larger inputs"
        )


def _validate_max_candidates(max_candidates: int) -> None:
    """Validate the response-size limit.

    Args:
        max_candidates: Candidate limit.

    Raises:
        ValueError: If the limit is invalid or exceeds the response ceiling.
    """
    _validate_at_most(
        max_candidates,
        MAX_RETURNED_CANDIDATES,
        "max_candidates",
    )


def _validate_beam_work(inputs: list[str], top_k: int, steps: int) -> None:
    """Reject beam-search requests with an excessive expansion upper bound.

    Args:
        inputs: Preprocessed strings passed to beam search.
        top_k: Beam width retained after each step.
        steps: Number of beam-search expansion steps.

    Raises:
        ValueError: If the request can exceed the process-wide work budget.
    """
    estimated_expansions = 0
    for value in inputs:
        length = len(value)
        estimated_expansions += length
        estimated_expansions += top_k * sum(
            length + step for step in range(1, steps)
        )
        if estimated_expansions > MAX_BEAM_EXPANSIONS:
            raise ValueError(
                "beam-search request is too large; reduce the input batch, "
                "top_k, or steps"
            )


def _validate_finite(value: float, name: str) -> None:
    """Validate a finite numeric value.

    Args:
        value: Value to validate.
        name: Argument name used in an error message.

    Raises:
        ValueError: If the value is boolean, nonnumeric, or non-finite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")


def _validate_hashtag_character(hashtag_character: str) -> None:
    """Validate a single-character hashtag marker.

    Args:
        hashtag_character: Marker used during preprocessing and replacement.

    Raises:
        ValueError: If the marker is not exactly one character.
    """
    if not isinstance(hashtag_character, str) or len(hashtag_character) != 1:
        raise ValueError("hashtag_character must contain exactly one character")


def _validate_strings(values: list[str], name: str) -> None:
    """Validate a list whose elements must all be strings.

    Args:
        values: Values to validate.
        name: Argument name used in an error message.

    Raises:
        ValueError: If any element is not a string.
    """
    if not isinstance(values, list) or any(
        not isinstance(value, str) for value in values
    ):
        raise ValueError(f"{name} must be a list of strings")


def _validate_regex_inputs(values: list[str]) -> None:
    """Validate bounded text passed to the timeout-protected regex engine.

    Args:
        values: Text values to validate.

    Raises:
        ValueError: If an input exceeds the regex safety ceiling.
    """
    for value in values:
        if len(value) > MAX_REGEX_INPUT_LENGTH:
            raise ValueError(
                f"regex inputs must contain at most {MAX_REGEX_INPUT_LENGTH} characters"
            )


def _validate_hashtag_lengths(values: list[str]) -> None:
    """Prevent pathological beam trees from unbounded input strings.

    Args:
        values: Hashtags or unspaced words to validate.

    Raises:
        ValueError: If an input exceeds the hashtag-length ceiling.
    """
    for value in values:
        if len(value) > MAX_HASHTAG_LENGTH:
            raise ValueError(
                f"hashtags must contain at most {MAX_HASHTAG_LENGTH} characters"
            )


def _compile_regex_rules(
    regex_rules: list[str] | None,
) -> list[timeout_regex.Pattern]:
    """Compile a bounded rule set for timeout-protected substitution.

    Args:
        regex_rules: Ordered custom rules or ``None`` for the library default.

    Returns:
        Compiled rules in caller order.

    Raises:
        ValueError: If the rule set exceeds MCP safety limits.
    """
    rules = regex_rules if regex_rules is not None else [r"([A-Z]+)"]
    _validate_strings(rules, "regex_rules")
    if not rules:
        raise ValueError("regex_rules must not be empty when provided")
    if len(rules) > MAX_REGEX_RULES:
        raise ValueError(f"regex_rules must contain at most {MAX_REGEX_RULES} rules")
    for rule in rules:
        if len(rule) > MAX_REGEX_PATTERN_LENGTH:
            raise ValueError(
                "regex rules must contain at most "
                f"{MAX_REGEX_PATTERN_LENGTH} characters"
            )
    try:
        return [timeout_regex.compile(rule) for rule in rules]
    except timeout_regex.error as error:
        raise ValueError(f"invalid regex rule: {error}") from error


class _TimeoutRegexWordSegmenter(RegexWordSegmenter):
    """Apply RegexWordSegmenter semantics with per-substitution timeouts.

    """

    def __init__(self, regex_rules: list[str] | None = None):
        """Compile the configured ordered regex rules.

        Args:
            regex_rules: Ordered custom rules or ``None`` for the default.
        """
        self.regex_rules = _compile_regex_rules(regex_rules)

    def segment_word(self, rule: timeout_regex.Pattern, word: str) -> str:
        """Apply one segmentation rule within a fixed CPU-time budget.

        Args:
            rule: Compiled timeout-capable rule.
            word: Current intermediate segmentation.

        Returns:
            Text after inserting the rule's captured word boundary.

        Raises:
            ValueError: If the rule times out or expands the intermediate too far.
        """
        try:
            segmented = rule.sub(
                r" \1",
                word,
                timeout=REGEX_TIMEOUT_SECONDS,
            ).strip()
        except TimeoutError as error:
            raise ValueError("regex rule exceeded the execution timeout") from error
        if len(segmented) > MAX_REGEX_OUTPUT_LENGTH:
            raise ValueError("regex rules produced an oversized intermediate result")
        return segmented


def _normalize_inputs(
    inputs: list[str],
    lower: bool,
    remove_hashtag: bool,
    hashtag_character: str,
) -> list[str]:
    """Apply the preprocessing implemented by ``BaseSegmenter``.

    Args:
        inputs: Input strings to normalize.
        lower: Whether to lowercase the inputs.
        remove_hashtag: Whether to strip leading hashtag characters.
        hashtag_character: Character stripped from the left edge.

    Returns:
        Normalized inputs in their original order.
    """
    normalized = []
    for value in inputs:
        if lower:
            value = value.lower()
        if remove_hashtag:
            value = value.lstrip(hashtag_character)
        normalized.append(value)
    return normalized


def _validate_runtime_options(
    top_k: int,
    steps: int,
    alpha: float,
    beta: float,
    max_candidates: int,
    hashtag_character: str,
) -> None:
    """Validate shared Transformer inference options.

    Args:
        top_k: Beam width.
        steps: Beam-search depth.
        alpha: Segmenter ensemble weight.
        beta: Reranker ensemble weight.
        max_candidates: Optional response-size limit.
        hashtag_character: Preprocessing marker.

    Raises:
        ValueError: If an option is outside its supported domain.
    """
    _validate_at_most(top_k, MAX_TOP_K, "top_k")
    _validate_at_most(steps, MAX_STEPS, "steps")
    _validate_finite(alpha, "alpha")
    _validate_finite(beta, "beta")
    _validate_max_candidates(max_candidates)
    _validate_hashtag_character(hashtag_character)


def _resolve_ranking_strategy(strategy: RankingStrategy) -> ResolvedRankingStrategy:
    """Resolve an explicit pipeline component for result selection.

    Args:
        strategy: Requested strategy or ``auto``.

    Returns:
        Concrete component used to select the result.

    Raises:
        ValueError: If the strategy is invalid or needs an absent reranker.
    """
    allowed = {"auto", "segmenter", "reranker", "ensemble"}
    if strategy not in allowed:
        raise ValueError(
            "ranking_strategy must be auto, segmenter, reranker, or ensemble"
        )
    if strategy == "auto":
        return "ensemble" if _server_config.reranker_model else "segmenter"
    if strategy in {"reranker", "ensemble"} and not _server_config.reranker_model:
        raise ValueError(
            f"ranking_strategy={strategy} requires --reranker-model at server startup"
        )
    return strategy


def _strategy_flags(
    strategy: ResolvedRankingStrategy,
) -> tuple[bool, bool]:
    """Map a ranking strategy to library execution flags.

    Args:
        strategy: Concrete ranking component.

    Returns:
        A ``use_reranker``, ``use_ensembler`` pair.
    """
    return strategy != "segmenter", strategy == "ensemble"


def _run_transformer_pipeline(
    segmenter: TransformerWordSegmenter,
    inputs: list[str],
    top_k: int,
    steps: int,
    alpha: float,
    beta: float,
    strategy: ResolvedRankingStrategy,
    preprocessing_kwargs: dict,
    segmenter_run=None,
) -> WordSegmenterOutput:
    """Run every public ``BaseWordSegmenter`` option on the configured models.

    Args:
        segmenter: Process-wide configured Transformer segmenter.
        inputs: Words or hashtags to segment or align with precomputed scores.
        top_k: Beam width retained after each search step.
        steps: Number of beam-search expansion steps.
        alpha: Segmenter-score ensemble weight.
        beta: Reranker-score ensemble weight.
        strategy: Component used for final selection.
        preprocessing_kwargs: Options forwarded to input preprocessing.
        segmenter_run: Optional precomputed candidate scores.

    Returns:
        Selected segmentations and every component ranking that ran.
    """
    use_reranker, use_ensembler = _strategy_flags(strategy)
    return BaseWordSegmenter.segment(
        segmenter,
        inputs,
        segmenter_run=segmenter_run,
        preprocessing_kwargs=preprocessing_kwargs,
        segmenter_kwargs={"topk": top_k, "steps": steps},
        ensembler_kwargs={"alpha": alpha, "beta": beta},
        use_reranker=use_reranker,
        use_ensembler=use_ensembler,
        return_ranks=True,
    )


def _serialize_rank(
    rank,
    normalized_input: str,
    max_candidates: int,
) -> list[Candidate]:
    """Serialize one component ranking for one input.

    Args:
        rank: Candidate ranking DataFrame returned by Hashformers.
        normalized_input: Preprocessed input used to select matching rows.
        max_candidates: Maximum rows to return.

    Returns:
        JSON-safe candidates ordered by ascending score.

    Raises:
        RuntimeError: If a model emits a non-finite score.
    """
    if rank is None:
        return []
    characters = normalized_input.replace(" ", "")
    rows = rank[rank["characters"] == characters]
    rows = rows.sort_values(["score", "segmentation"], kind="stable")
    rows = rows.head(max_candidates)
    candidates = []
    for position, row in enumerate(rows.itertuples(index=False), start=1):
        score = float(row.score)
        if not math.isfinite(score):
            raise RuntimeError("Hashformers produced a non-finite candidate score")
        candidates.append(
            {
                "segmentation": str(row.segmentation),
                "score": score,
                "rank": position,
            }
        )
    return candidates


def _selected_rank(output: WordSegmenterOutput, strategy: ResolvedRankingStrategy):
    """Select the ranking table corresponding to the requested strategy.

    Args:
        output: Ranked Hashformers output.
        strategy: Component that selected the result.

    Returns:
        The selected component ranking table.
    """
    if strategy == "ensemble":
        return output.ensemble_rank
    if strategy == "reranker":
        return output.reranker_rank
    return output.segmenter_rank


def _serialize_word_output(
    original_inputs: list[str],
    normalized_inputs: list[str],
    output: WordSegmenterOutput,
    strategy: ResolvedRankingStrategy,
    max_candidates: int,
    include_component_rankings: bool,
) -> list[SegmentationResult]:
    """Serialize a ``WordSegmenterOutput`` for MCP.

    Args:
        original_inputs: Caller-provided inputs.
        normalized_inputs: Inputs after preprocessing.
        output: Ranked output from Hashformers.
        strategy: Component that selected each result.
        max_candidates: Maximum candidates returned per component.
        include_component_rankings: Whether to include every component rank.

    Returns:
        Per-input JSON-safe segmentation results.

    Raises:
        RuntimeError: If a selected component produced no matching candidates.
    """
    if len(output.output) != len(original_inputs):
        raise RuntimeError(
            "Hashformers returned a different number of results than inputs"
        )
    selected_rank = _selected_rank(output, strategy)
    results = []
    for original, normalized, selected in zip(
        original_inputs,
        normalized_inputs,
        output.output,
    ):
        candidates = _serialize_rank(selected_rank, normalized, max_candidates)
        if not candidates:
            raise RuntimeError(f"Hashformers produced no candidates for {original!r}")
        component_rankings = None
        if include_component_rankings:
            component_rankings = {
                "segmenter": _serialize_rank(
                    output.segmenter_rank,
                    normalized,
                    max_candidates,
                ),
                "reranker": (
                    _serialize_rank(output.reranker_rank, normalized, max_candidates)
                    if output.reranker_rank is not None
                    else None
                ),
                "ensemble": (
                    _serialize_rank(output.ensemble_rank, normalized, max_candidates)
                    if output.ensemble_rank is not None
                    else None
                ),
            }
        results.append(
            {
                "input": original,
                "normalized_input": normalized,
                "selected_segmentation": str(selected),
                "ranking_strategy": strategy,
                "candidates": candidates,
                "component_rankings": component_rankings,
            }
        )
    return results


def _regex_result(
    original_input: str,
    normalized_input: str,
    segmentation: str,
) -> SegmentationResult:
    """Build a segmentation result for a deterministic regex output.

    Args:
        original_input: Text before preprocessing.
        normalized_input: Text after preprocessing.
        segmentation: Regex segmentation.

    Returns:
        A result compatible with Transformer hashtag results.
    """
    candidate: Candidate = {"segmentation": segmentation, "score": 0.0, "rank": 1}
    return {
        "input": original_input,
        "normalized_input": normalized_input,
        "selected_segmentation": segmentation,
        "ranking_strategy": "segmenter",
        "candidates": [candidate],
        "component_rankings": None,
    }


def _resolve_input_format(
    input_path: Path,
    input_format: FileInputFormat,
) -> Literal["text", "csv", "jsonl"]:
    """Resolve automatic file-format selection from a filename suffix.

    Args:
        input_path: File to inspect by name.
        input_format: Explicit format or ``auto``.

    Returns:
        Concrete supported input format.

    Raises:
        ValueError: If the format is unsupported.
    """
    allowed = {"auto", "text", "csv", "jsonl"}
    if input_format not in allowed:
        raise ValueError("input_format must be auto, text, csv, or jsonl")
    if input_format != "auto":
        return input_format
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    return "text"


class _BoundedTextLines:
    """Iterate over bounded physical and logical text-file records.

    Args:
        file_object: Open text stream consumed one physical line at a time.
    """

    def __init__(self, file_object: TextIO):
        """Initialize a bounded line iterator.

        Args:
            file_object: Open text stream to consume.
        """
        self.file_object = file_object
        self.record_chars = 0

    def __iter__(self):
        """Return the iterator itself.

        Returns:
            This bounded line iterator.
        """
        return self

    def __next__(self) -> str:
        """Read one bounded physical line.

        Returns:
            The next physical line from the input stream.

        Raises:
            ValueError: If a physical or logical record exceeds the ceiling.
            StopIteration: If the input stream is exhausted.
        """
        line = self.file_object.readline(MAX_FILE_RECORD_CHARS + 1)
        if not line:
            raise StopIteration
        if len(line) > MAX_FILE_RECORD_CHARS:
            raise ValueError(
                f"file records must contain at most {MAX_FILE_RECORD_CHARS} "
                "characters"
            )
        self.record_chars += len(line)
        if self.record_chars > MAX_FILE_RECORD_CHARS:
            raise ValueError(
                f"file records must contain at most {MAX_FILE_RECORD_CHARS} "
                "characters"
            )
        return line

    def reset_record(self) -> None:
        """Start accounting for the next logical record.

        Returns:
            None.
        """
        self.record_chars = 0


def _iter_file_hashtags(
    input_source: Path | TextIO,
    input_format: Literal["text", "csv", "jsonl"],
    input_field: str,
) -> Iterator[tuple[int, str]]:
    """Stream hashtags from text, CSV, or JSON Lines input.

    Args:
        input_source: UTF-8 path or already-open text stream.
        input_format: Concrete file format.
        input_field: Object or CSV column containing each hashtag.

    Yields:
        Source line numbers and hashtags in file order.

    Raises:
        ValueError: If a record is blank, malformed, or lacks ``input_field``.
    """
    if isinstance(input_source, Path):
        with input_source.open("r", encoding="utf-8", newline="") as input_file:
            yield from _iter_file_hashtags(input_file, input_format, input_field)
        return

    lines = _BoundedTextLines(input_source)
    if input_format == "text":
        for line_number, line in enumerate(lines, start=1):
            lines.reset_record()
            hashtag = line.rstrip("\r\n")
            if not isinstance(hashtag, str) or not hashtag.strip():
                raise ValueError(f"blank hashtag at line {line_number}")
            yield line_number, hashtag
        return

    if input_format == "csv":
        reader = csv.DictReader(lines)
        if reader.fieldnames is None or input_field not in reader.fieldnames:
            raise ValueError(f"CSV input must contain column {input_field!r}")
        lines.reset_record()
        for line_number, record in enumerate(reader, start=2):
            lines.reset_record()
            hashtag = record.get(input_field)
            if not isinstance(hashtag, str) or not hashtag.strip():
                raise ValueError(f"blank hashtag at CSV line {line_number}")
            yield line_number, hashtag
        return

    for line_number, line in enumerate(lines, start=1):
        lines.reset_record()
        if not line.strip():
            raise ValueError(f"blank JSON Lines record at line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON Lines record at line {line_number}"
            ) from error
        if isinstance(record, str):
            hashtag = record
        elif isinstance(record, dict):
            hashtag = record.get(input_field)
        else:
            hashtag = None
        if not isinstance(hashtag, str) or not hashtag.strip():
            raise ValueError(
                f"JSON Lines record {line_number} must contain "
                f"string field {input_field!r}"
            )
        yield line_number, hashtag


def _hash_descriptor(descriptor: int) -> str:
    """Calculate SHA-256 from an already-open regular file descriptor.

    Args:
        descriptor: Readable descriptor whose offset may be changed.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _validate_job_schema(connection: sqlite3.Connection) -> None:
    """Require the exact bounded tables created by Hashformers.

    Args:
        connection: Open candidate checkpoint database.

    Returns:
        None.

    Raises:
        ValueError: If any required table or constraint differs.
    """
    for table_name, expected in JOB_TABLE_SCHEMAS.items():
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        actual = tuple(
            (name, column_type.upper(), not_null, primary_key)
            for _, name, column_type, not_null, _, primary_key in rows
        )
        if actual != expected:
            raise ValueError("job_path is not a valid Hashformers checkpoint")


def _job_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    """Read all metadata from a file-job checkpoint.

    Args:
        connection: Open job database.

    Returns:
        Metadata keyed by field name.

    Raises:
        ValueError: If the database is not a bounded Hashformers checkpoint.
    """
    application_id = connection.execute("PRAGMA application_id").fetchone()[0]
    schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if (
        application_id != JOB_APPLICATION_ID
        or schema_version != JOB_SCHEMA_VERSION
    ):
        raise ValueError("job_path is not a valid Hashformers checkpoint")
    _validate_job_schema(connection)
    placeholders = ", ".join("?" for _ in JOB_METADATA_KEYS)
    rows = connection.execute(
        f"""
        SELECT key, substr(value, 1, ?)
        FROM metadata
        WHERE key IN ({placeholders})
        LIMIT ?
        """,
        (
            MAX_JOB_METADATA_CHARS + 1,
            *JOB_METADATA_KEYS,
            len(JOB_METADATA_KEYS) + 1,
        ),
    ).fetchall()
    if (
        len(rows) != len(JOB_METADATA_KEYS)
        or {key for key, _ in rows} != set(JOB_METADATA_KEYS)
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or len(value) > MAX_JOB_METADATA_CHARS
            for key, value in rows
        )
    ):
        raise ValueError("job checkpoint metadata is invalid or oversized")
    return dict(rows)


def _job_counts(connection: sqlite3.Connection) -> tuple[int, int, int]:
    """Count source, unique, and completed records in a job.

    Args:
        connection: Open job database.

    Returns:
        Total source records, unique inputs, and completed unique inputs.
    """
    total = connection.execute("SELECT COUNT(*) FROM source_records").fetchone()[0]
    unique = connection.execute("SELECT COUNT(*) FROM unique_hashtags").fetchone()[0]
    processed = connection.execute(
        "SELECT COUNT(*) FROM unique_hashtags WHERE result_json IS NOT NULL"
    ).fetchone()[0]
    return total, unique, processed


def _file_job_status(
    connection: sqlite3.Connection,
    job_path: Path,
    processed_this_call: int = 0,
) -> FileJobStatus:
    """Build the compact MCP response for a file job.

    Args:
        connection: Open job database.
        job_path: Persistent checkpoint path.
        processed_this_call: Unique inputs completed by the current call.

    Returns:
        JSON-safe job progress and reproducibility metadata.
    """
    metadata = _job_metadata(connection)
    total, unique, processed = _job_counts(connection)
    remaining = unique - processed
    server_config = json.loads(metadata["server_config"])
    run_options = json.loads(metadata["run_options"])
    return {
        "job_path": str(job_path),
        "input_path": metadata["input_path"],
        "output_path": metadata["output_path"],
        "input_format": metadata["input_format"],
        "status": "completed" if remaining == 0 else "in_progress",
        "total_hashtags": total,
        "unique_hashtags": unique,
        "deduplicated_hashtags": total - unique,
        "processed_unique": processed,
        "processed_this_call": processed_this_call,
        "remaining_unique": remaining,
        "segmenter_model": server_config["segmenter_model"],
        "reranker_model": server_config["reranker_model"],
        "ranking_strategy": run_options["ranking_strategy"],
    }


def _finalize_file_job(
    connection: sqlite3.Connection,
    destination: Path,
    overwrite: bool,
) -> None:
    """Materialize checkpointed results in original source order.

    Args:
        connection: Completed job database.
        destination: JSON Lines output path.
        overwrite: Whether an existing destination may be replaced.

    Returns:
        None.

    Raises:
        ValueError: If output exists without overwrite authorization.
    """
    destination, parent_descriptor = _open_authorized_parent(
        destination,
        "output_path",
        must_exist=False,
    )
    temporary_name = None
    try:
        temporary_name = (
            f".{destination.name}.{secrets.token_hex(16)}.tmp"
        )
        temporary_descriptor = _open_child(
            parent_descriptor,
            destination.parent,
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(
            temporary_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            rows = connection.execute(
                """
                SELECT CASE
                           WHEN typeof(source_records.source_line) = 'integer'
                           THEN source_records.source_line
                       END,
                       substr(source_records.input, 1, ?),
                       substr(unique_hashtags.result_json, 1, ?)
                FROM source_records
                JOIN unique_hashtags USING (normalized)
                ORDER BY source_records.rowid
                """,
                (
                    MAX_HASHTAG_LENGTH + 1,
                    MAX_JOB_RESULT_CHARS + 1,
                ),
            )
            for source_line, original_input, result_json in rows:
                if (
                    isinstance(source_line, bool)
                    or not isinstance(source_line, int)
                    or source_line < 1
                ):
                    raise ValueError(
                        "job checkpoint contains an invalid source line"
                    )
                if (
                    not isinstance(original_input, str)
                    or len(original_input) > MAX_HASHTAG_LENGTH
                ):
                    raise ValueError(
                        "job checkpoint contains an oversized source input"
                    )
                if (
                    not isinstance(result_json, str)
                    or len(result_json) > MAX_JOB_RESULT_CHARS
                ):
                    raise ValueError(
                        "job checkpoint contains an oversized result record"
                    )
                record = json.loads(result_json)
                record["input"] = original_input
                output_file.write(
                    json.dumps(
                        {"source_line": source_line, **record},
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
                output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        if overwrite:
            _replace_child(
                parent_descriptor,
                destination.parent,
                temporary_name,
                destination.name,
            )
            temporary_name = None
        else:
            try:
                _link_children(
                    parent_descriptor,
                    destination.parent,
                    temporary_name,
                    destination.name,
                )
            except FileExistsError as error:
                existing_descriptor = None
                temporary_descriptor = None
                try:
                    existing_descriptor = _open_regular_child(
                        parent_descriptor,
                        destination.parent,
                        destination.name,
                        os.O_RDONLY,
                        "output_path must be a regular file inside its "
                        "configured root",
                    )
                    temporary_descriptor = _open_regular_child(
                        parent_descriptor,
                        destination.parent,
                        temporary_name,
                        os.O_RDONLY,
                        "temporary output is not a regular file",
                    )
                    if _hash_descriptor(existing_descriptor) == _hash_descriptor(
                        temporary_descriptor
                    ):
                        _unlink_child(
                            parent_descriptor,
                            destination.parent,
                            temporary_name,
                        )
                        temporary_name = None
                        _fsync_parent(parent_descriptor, destination.parent)
                        return
                except OSError as open_error:
                    raise ValueError(
                        "output_path must be a regular file inside its "
                        "configured root"
                    ) from open_error
                finally:
                    if existing_descriptor is not None:
                        os.close(existing_descriptor)
                    if temporary_descriptor is not None:
                        os.close(temporary_descriptor)
                if temporary_name is None:
                    return
                raise ValueError(
                    "output_path contains different content and overwrite is "
                    f"disabled: {destination}"
                ) from error
            _unlink_child(
                parent_descriptor,
                destination.parent,
                temporary_name,
            )
            temporary_name = None
        _fsync_parent(parent_descriptor, destination.parent)
    finally:
        if temporary_name is not None:
            try:
                _unlink_child(
                    parent_descriptor,
                    destination.parent,
                    temporary_name,
                )
            except FileNotFoundError:
                pass
        if parent_descriptor is not None:
            os.close(parent_descriptor)


mcp = MCPServer(
    "hashformers",
    description=(
        "Segment hashtags and tweets, apply regex segmentation, and rank "
        "precomputed hypotheses with Hashformers."
    ),
)


MODEL_READ_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)
LOCAL_READ_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
FILE_WRITE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
)
FILE_MODEL_WRITE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)


@mcp.tool(annotations=MODEL_READ_ANNOTATIONS)
def segment_hashtags(
    hashtags: list[str],
    top_k: int = 5,
    steps: int = 5,
    ranking_strategy: RankingStrategy = "auto",
    alpha: float = 0.222,
    beta: float = 0.111,
    lower: bool = False,
    remove_hashtag: bool = True,
    hashtag_character: str = "#",
    max_candidates: int = 5,
    include_component_rankings: bool = False,
) -> SegmentationsResult:
    """Segment hashtags with every option exposed by the Transformer pipeline.

    Args:
        hashtags: Hashtags or unspaced words to segment.
        top_k: Beam width retained after each search step.
        steps: Number of beam-search expansion steps.
        ranking_strategy: Component used to select the final segmentation.
        alpha: Segmenter-score weight used by the ensemble.
        beta: Reranker-score weight used by the ensemble.
        lower: Whether to lowercase inputs before segmentation.
        remove_hashtag: Whether to strip leading hashtag characters.
        hashtag_character: Character stripped during preprocessing.
        max_candidates: Response limit per ranking, capped at 64.
        include_component_rankings: Whether to return every component rank.

    Returns:
        Selected segmentations, final candidates, and optional component ranks.

    Raises:
        ValueError: If inputs or options are invalid.
    """
    _validate_strings(hashtags, "hashtags")
    _validate_input_count(hashtags, "hashtags")
    _validate_hashtag_lengths(hashtags)
    _validate_runtime_options(
        top_k,
        steps,
        alpha,
        beta,
        max_candidates,
        hashtag_character,
    )
    strategy = _resolve_ranking_strategy(ranking_strategy)
    normalized = _normalize_inputs(
        hashtags,
        lower=lower,
        remove_hashtag=remove_hashtag,
        hashtag_character=hashtag_character,
    )
    if any(not value.strip() for value in normalized):
        raise ValueError("hashtags must contain text after preprocessing")
    if not hashtags:
        return {"results": []}
    _validate_beam_work(normalized, top_k, steps)

    with _inference_lock:
        output = _run_transformer_pipeline(
            get_segmenter(),
            hashtags,
            top_k=top_k,
            steps=steps,
            alpha=alpha,
            beta=beta,
            strategy=strategy,
            preprocessing_kwargs={
                "lower": lower,
                "remove_hashtag": remove_hashtag,
                "hashtag_character": hashtag_character,
            },
        )
    return {
        "results": _serialize_word_output(
            hashtags,
            normalized,
            output,
            strategy,
            max_candidates,
            include_component_rankings,
        )
    }


@mcp.tool(annotations=FILE_WRITE_ANNOTATIONS)
def start_hashtag_file_job(
    input_path: str,
    output_path: str | None = None,
    input_format: FileInputFormat = "auto",
    input_field: str = "hashtag",
    overwrite: bool = False,
    top_k: int = 5,
    steps: int = 5,
    ranking_strategy: RankingStrategy = "auto",
    alpha: float = 0.222,
    beta: float = 0.111,
    lower: bool = False,
    remove_hashtag: bool = True,
    hashtag_character: str = "#",
    max_candidates: int = 1,
    include_component_rankings: bool = False,
) -> FileJobStatus:
    """Index and checkpoint a local hashtag file without model inference.

    Args:
        input_path: UTF-8 text, CSV, or JSON Lines file read by the server.
        output_path: Final JSON Lines destination, or a derived sibling path.
        input_format: Input format or ``auto`` for suffix-based detection.
        input_field: CSV column or JSON object field containing hashtags.
        overwrite: Whether an existing output may be atomically replaced.
        top_k: Beam width retained after each search step.
        steps: Number of beam-search expansion steps.
        ranking_strategy: Component used to select final segmentations.
        alpha: Segmenter-score weight used by the ensemble.
        beta: Reranker-score weight used by the ensemble.
        lower: Whether to lowercase inputs before segmentation.
        remove_hashtag: Whether to strip leading hashtag characters.
        hashtag_character: Character stripped during preprocessing.
        max_candidates: Candidate limit written per result, capped at 64.
        include_component_rankings: Whether to write every component rank.

    Returns:
        A persistent job path and compact progress counts.

    Raises:
        ValueError: If paths, records, or segmentation options are invalid.
    """
    if not isinstance(input_path, str) or not input_path.strip():
        raise ValueError("input_path must contain text")
    if output_path is not None and (
        not isinstance(output_path, str) or not output_path.strip()
    ):
        raise ValueError("output_path must contain text or be null")
    if not isinstance(input_field, str) or not input_field.strip():
        raise ValueError("input_field must contain text")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")
    if overwrite and not _server_config.allow_file_overwrite:
        raise ValueError(
            "overwrite=true requires --allow-file-overwrite at server startup"
        )
    _validate_runtime_options(
        top_k,
        steps,
        alpha,
        beta,
        max_candidates,
        hashtag_character,
    )
    resolved_strategy = _resolve_ranking_strategy(ranking_strategy)

    source = _resolve_file_path(input_path, "input_path", must_exist=True)
    if not source.is_file():
        raise ValueError(f"input_path is not a readable file: {source}")
    resolved_format = _resolve_input_format(source, input_format)
    destination = (
        _resolve_file_path(output_path, "output_path", must_exist=False)
        if output_path is not None
        else source.with_name(f"{source.name}.hashformers.jsonl")
    )
    job_path = _resolve_file_path(
        f"{destination}.job.sqlite3",
        "job_path",
        must_exist=False,
    )
    if source in {destination, job_path}:
        raise ValueError("input, output, and job paths must be different")
    if not destination.parent.is_dir():
        raise ValueError(f"output directory does not exist: {destination.parent}")
    if destination.exists() and not overwrite:
        raise ValueError(
            "output_path already exists; pass overwrite=true to replace it: "
            f"{destination}"
        )
    if job_path.exists():
        raise ValueError(
            f"job checkpoint already exists; continue it instead: {job_path}"
        )

    run_options = {
        "top_k": top_k,
        "steps": steps,
        "ranking_strategy": resolved_strategy,
        "alpha": float(alpha),
        "beta": float(beta),
        "lower": lower,
        "remove_hashtag": remove_hashtag,
        "hashtag_character": hashtag_character,
        "max_candidates": max_candidates,
        "include_component_rankings": include_component_rankings,
    }
    source_descriptor = None
    source_parent_descriptor = None
    job_parent_descriptor = None
    temporary_descriptor = None
    temporary_name = None
    connection = None
    try:
        source, source_parent_descriptor = _open_authorized_parent(
            source,
            "input_path",
            must_exist=True,
        )
        source_descriptor = _open_regular_child(
            source_parent_descriptor,
            source.parent,
            source.name,
            os.O_RDONLY,
            f"input_path is not a readable file: {source}",
        )
        initial_hash = _hash_descriptor(source_descriptor)

        job_path, job_parent_descriptor = _open_authorized_parent(
            job_path,
            "job_path",
            must_exist=False,
        )
        temporary_name = f".{job_path.name}.{secrets.token_hex(16)}.tmp"
        temporary_descriptor = _open_child(
            job_parent_descriptor,
            job_path.parent,
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        connection = _connect_descriptor_database(temporary_descriptor)
        connection.executescript(
            f"""
            PRAGMA application_id = {JOB_APPLICATION_ID};
            PRAGMA user_version = {JOB_SCHEMA_VERSION};
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE source_records (
                source_line INTEGER PRIMARY KEY,
                input TEXT NOT NULL,
                normalized TEXT NOT NULL
            );
            CREATE TABLE unique_hashtags (
                normalized TEXT PRIMARY KEY,
                representative_input TEXT NOT NULL,
                result_json TEXT
            );
            """
        )
        metadata = {
            "input_path": str(source),
            "output_path": str(destination),
            "input_format": resolved_format,
            "input_field": input_field,
            "input_hash": initial_hash,
            "overwrite": json.dumps(overwrite),
            "finalized": json.dumps(False),
            "server_config": json.dumps(_model_config_payload(), sort_keys=True),
            "run_options": json.dumps(run_options, sort_keys=True),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        source_stream = os.fdopen(
            os.dup(source_descriptor),
            "r",
            encoding="utf-8",
            newline="",
        )
        with source_stream:
            for source_line, hashtag in _iter_file_hashtags(
                source_stream,
                resolved_format,
                input_field,
            ):
                _validate_hashtag_lengths([hashtag])
                normalized = _normalize_inputs(
                    [hashtag],
                    lower=lower,
                    remove_hashtag=remove_hashtag,
                    hashtag_character=hashtag_character,
                )[0]
                if not normalized.strip():
                    raise ValueError(
                        f"hashtag at source line {source_line} has no text after "
                        "preprocessing"
                    )
                _validate_beam_work([normalized], top_k, steps)
                connection.execute(
                    "INSERT INTO source_records VALUES (?, ?, ?)",
                    (source_line, hashtag, normalized),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO unique_hashtags VALUES (?, ?, NULL)",
                    (normalized, hashtag),
                )
        if _hash_descriptor(source_descriptor) != initial_hash:
            raise ValueError("input file changed while the job was being indexed")
        connection.commit()
        connection.close()
        connection = None
        os.fsync(temporary_descriptor)
        visible_temporary_descriptor = _open_regular_child(
            job_parent_descriptor,
            job_path.parent,
            temporary_name,
            os.O_RDONLY,
            "temporary checkpoint is not a regular file",
        )
        try:
            if not os.path.samestat(
                os.fstat(temporary_descriptor),
                os.fstat(visible_temporary_descriptor),
            ):
                raise ValueError("temporary checkpoint changed during indexing")
        finally:
            os.close(visible_temporary_descriptor)
        try:
            _link_children(
                job_parent_descriptor,
                job_path.parent,
                temporary_name,
                job_path.name,
            )
        except FileExistsError as error:
            raise ValueError(
                "job checkpoint was created concurrently; continue it instead: "
                f"{job_path}"
            ) from error
        _unlink_child(
            job_parent_descriptor,
            job_path.parent,
            temporary_name,
        )
        temporary_name = None
        _fsync_parent(job_parent_descriptor, job_path.parent)

        with _connect_checkpoint(job_path, timeout=0) as installed_connection:
            status = _file_job_status(installed_connection, job_path)
            if status["status"] == "completed":
                _finalize_file_job(installed_connection, destination, overwrite)
                installed_connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'finalized'",
                    (json.dumps(True),),
                )
                installed_connection.commit()
        return status
    finally:
        if connection is not None:
            connection.close()
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if temporary_name is not None:
            try:
                _unlink_child(
                    job_parent_descriptor,
                    job_path.parent,
                    temporary_name,
                )
            except FileNotFoundError:
                pass
        if source_descriptor is not None:
            os.close(source_descriptor)
        if source_parent_descriptor is not None:
            os.close(source_parent_descriptor)
        if job_parent_descriptor is not None:
            os.close(job_parent_descriptor)


def _continue_file_job_sync(
    checkpoint: Path,
    max_unique_hashtags: int,
) -> FileJobStatus:
    """Claim, infer, persist, and finalize one file-job chunk in a worker.

    Args:
        checkpoint: Authorized SQLite checkpoint path.
        max_unique_hashtags: Maximum unique inputs processed in this chunk.

    Returns:
        Updated persistent job status.

    Raises:
        ValueError: If the checkpoint is invalid, busy, or incompatible.
    """
    connection_manager = _connect_checkpoint(checkpoint, timeout=0)
    connection = connection_manager.__enter__()
    try:
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as error:
            if "locked" in str(error).lower() or "busy" in str(error).lower():
                raise ValueError("job is already being processed") from error
            raise
        try:
            metadata = _job_metadata(connection)
        except sqlite3.DatabaseError as error:
            raise ValueError(
                "job_path is not a valid Hashformers checkpoint"
            ) from error
        required_metadata = set(JOB_METADATA_KEYS)
        if not required_metadata.issubset(metadata):
            raise ValueError("job_path is not a valid Hashformers checkpoint")
        _resolve_file_path(metadata["input_path"], "input_path", must_exist=False)
        destination = _resolve_file_path(
            metadata["output_path"],
            "output_path",
            must_exist=False,
        )
        overwrite = json.loads(metadata["overwrite"])
        if not isinstance(overwrite, bool):
            raise ValueError("job checkpoint contains an invalid overwrite policy")
        if overwrite and not _server_config.allow_file_overwrite:
            raise ValueError(
                "job requires --allow-file-overwrite at server startup"
            )
        active_config = json.dumps(_model_config_payload(), sort_keys=True)
        if active_config != metadata["server_config"]:
            raise ValueError(
                "MCP model configuration changed since the job was started"
            )
        run_options = json.loads(metadata["run_options"])
        pending_rows = connection.execute(
            """
            SELECT substr(normalized, 1, ?),
                   substr(representative_input, 1, ?)
            FROM unique_hashtags
            WHERE result_json IS NULL
            ORDER BY rowid
            LIMIT ?
            """,
            (
                MAX_HASHTAG_LENGTH + 1,
                MAX_HASHTAG_LENGTH + 1,
                max_unique_hashtags,
            ),
        ).fetchall()
        if any(
            not isinstance(normalized, str)
            or not isinstance(representative, str)
            or len(normalized) > MAX_HASHTAG_LENGTH
            or len(representative) > MAX_HASHTAG_LENGTH
            for normalized, representative in pending_rows
        ):
            raise ValueError("job checkpoint contains an oversized hashtag")
        pending = pending_rows
        if pending:
            representatives = [representative for _, representative in pending]
            result = segment_hashtags(representatives, **run_options)
            records_by_normalized = {
                record["normalized_input"]: record for record in result["results"]
            }
            for normalized, _ in pending:
                if normalized not in records_by_normalized:
                    raise RuntimeError(
                        f"Hashformers produced no result for {normalized!r}"
                    )
                connection.execute(
                    "UPDATE unique_hashtags SET result_json = ? WHERE normalized = ?",
                    (
                        json.dumps(
                            records_by_normalized[normalized],
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        normalized,
                    ),
                )

        # Make model results durable before publishing the final output. A
        # retry can then finish or reconcile publication without rerunning it.
        connection.commit()
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as error:
            if "locked" in str(error).lower() or "busy" in str(error).lower():
                raise ValueError("job is already being processed") from error
            raise
        metadata = _job_metadata(connection)
        status = _file_job_status(
            connection,
            checkpoint,
            processed_this_call=len(pending),
        )
        if status["status"] == "completed" and not json.loads(metadata["finalized"]):
            _finalize_file_job(connection, destination, overwrite)
            try:
                connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'finalized'",
                    (json.dumps(True),),
                )
                connection.commit()
            except sqlite3.OperationalError as error:
                raise ValueError(
                    "job checkpoint directory changed during processing"
                ) from error
        else:
            connection.commit()
        return status
    finally:
        connection_manager.__exit__(None, None, None)


@mcp.tool(annotations=FILE_MODEL_WRITE_ANNOTATIONS)
async def continue_hashtag_file_job(
    job_path: str,
    max_unique_hashtags: int = 64,
    context: Context | None = None,
) -> FileJobStatus:
    """Process one resumable, bounded chunk of a hashtag file job.

    Args:
        job_path: Checkpoint returned by ``start_hashtag_file_job``.
        max_unique_hashtags: Maximum unique hashtags inferred in this call.
        context: MCP request context used for progress and cancellation.

    Returns:
        Updated progress; repeat until ``status`` is ``completed``.

    Raises:
        ValueError: If the checkpoint, input, or active model configuration changed.
    """
    if not isinstance(job_path, str) or not job_path.strip():
        raise ValueError("job_path must contain text")
    _validate_at_most(
        max_unique_hashtags,
        MAX_FILE_UNIQUE_HASHTAGS,
        "max_unique_hashtags",
    )
    checkpoint = _resolve_file_path(job_path, "job_path", must_exist=True)
    if not checkpoint.is_file():
        raise ValueError(f"job_path is not a readable checkpoint: {checkpoint}")

    if context is not None:
        try:
            with _connect_checkpoint(checkpoint, timeout=5) as connection:
                _job_metadata(connection)
                _, total_unique, processed_before = _job_counts(connection)
        except sqlite3.DatabaseError as error:
            raise ValueError(
                "job_path is not a valid Hashformers checkpoint"
            ) from error
        await context.report_progress(
            processed_before,
            total_unique,
            "Continuing Hashformers file segmentation",
        )
    status = await anyio.to_thread.run_sync(
        lambda: _continue_file_job_sync(checkpoint, max_unique_hashtags),
        abandon_on_cancel=False,
    )
    if context is not None:
        await context.report_progress(
            status["processed_unique"],
            status["unique_hashtags"],
            f"Checkpointed {status['processed_unique']} unique hashtags",
        )
    return status


@mcp.tool(annotations=LOCAL_READ_ANNOTATIONS)
def segment_with_regex(
    inputs: list[str],
    regex_rules: list[str] | None = None,
    lower: bool = False,
    remove_hashtag: bool = True,
    hashtag_character: str = "#",
) -> RegexSegmentationsResult:
    """Segment text by applying regular-expression rules sequentially.

    Args:
        inputs: Texts to segment.
        regex_rules: Ordered rules, or ``None`` for the library default.
        lower: Whether to lowercase inputs before applying rules.
        remove_hashtag: Whether to strip leading hashtag characters.
        hashtag_character: Character stripped during preprocessing.

    Returns:
        Deterministic regex segmentations in input order.

    Raises:
        ValueError: If inputs, rules, or preprocessing options are invalid.
    """
    _validate_strings(inputs, "inputs")
    _validate_input_count(inputs, "inputs")
    _validate_regex_inputs(inputs)
    _validate_hashtag_character(hashtag_character)
    normalized = _normalize_inputs(
        inputs,
        lower=lower,
        remove_hashtag=remove_hashtag,
        hashtag_character=hashtag_character,
    )
    segmenter = _TimeoutRegexWordSegmenter(regex_rules=regex_rules)
    segmentations = segmenter.segment(
        inputs,
        lower=lower,
        remove_hashtag=remove_hashtag,
        hashtag_character=hashtag_character,
    )
    return {
        "results": [
            {
                "input": original,
                "normalized_input": normalized_input,
                "segmentation": segmentation,
            }
            for original, normalized_input, segmentation in zip(
                inputs,
                normalized,
                segmentations,
            )
        ]
    }


@mcp.tool(annotations=MODEL_READ_ANNOTATIONS)
def segment_tweets(
    tweets: list[str],
    segmenter_kind: SegmenterKind = "regex",
    regex_rules: list[str] | None = None,
    top_k: int = 5,
    steps: int = 5,
    ranking_strategy: RankingStrategy = "auto",
    alpha: float = 0.222,
    beta: float = 0.111,
    word_lower: bool = False,
    max_candidates: int = 5,
    include_component_rankings: bool = False,
    hashtag_token: str | None = None,
    lower: bool = False,
    separator: str = " ",
    hashtag_character: str = "#",
    regex_flags: int = 0,
) -> TweetSegmentationsResult:
    """Extract and segment hashtags inside complete tweets.

    Args:
        tweets: Tweet texts to transform.
        segmenter_kind: ``regex`` or the configured ``transformer`` pipeline.
        regex_rules: Ordered rules used by the regex segmenter.
        top_k: Transformer beam width.
        steps: Transformer beam-search depth.
        ranking_strategy: Transformer component used for final selection.
        alpha: Segmenter-score weight used by the ensemble.
        beta: Reranker-score weight used by the ensemble.
        word_lower: Whether to lowercase hashtags before segmentation.
        max_candidates: Transformer response limit, capped at 64.
        include_component_rankings: Whether to include Transformer component ranks.
        hashtag_token: Optional token prepended to each replacement.
        lower: Whether to lowercase segmented hashtag replacements.
        separator: Text placed between ``hashtag_token`` and a replacement.
        hashtag_character: Character used to identify replacement keys.
        regex_flags: Flags passed to Python regex replacement.

    Returns:
        Transformed tweets and per-occurrence hashtag segmentation details.

    Raises:
        ValueError: If inputs or selected-segmenter options are invalid.
    """
    _validate_strings(tweets, "tweets")
    _validate_input_count(tweets, "tweets")
    _validate_regex_inputs(tweets)
    _validate_hashtag_character(hashtag_character)
    if hashtag_character != "#":
        raise ValueError("segment_tweets supports only hashtag_character='#'")
    if segmenter_kind not in {"regex", "transformer"}:
        raise ValueError("segmenter_kind must be regex or transformer")
    if (
        isinstance(regex_flags, bool)
        or not isinstance(regex_flags, int)
        or regex_flags < 0
    ):
        raise ValueError("regex_flags must be a non-negative integer")
    if not isinstance(separator, str):
        raise ValueError("separator must be a string")
    if len(separator) > MAX_TWEET_REPLACEMENT_CHARS:
        raise ValueError(
            f"separator must contain at most {MAX_TWEET_REPLACEMENT_CHARS} "
            "characters"
        )
    if hashtag_token is not None and not isinstance(hashtag_token, str):
        raise ValueError("hashtag_token must be a string or null")
    if (
        hashtag_token is not None
        and len(hashtag_token) > MAX_TWEET_REPLACEMENT_CHARS
    ):
        raise ValueError(
            f"hashtag_token must contain at most {MAX_TWEET_REPLACEMENT_CHARS} "
            "characters"
        )
    if segmenter_kind == "transformer" and regex_rules is not None:
        raise ValueError("regex_rules are supported only with segmenter_kind='regex'")
    if segmenter_kind == "regex":
        _compile_regex_rules(regex_rules)
    if segmenter_kind == "transformer":
        _validate_runtime_options(
            top_k,
            steps,
            alpha,
            beta,
            max_candidates,
            hashtag_character,
        )
        strategy = _resolve_ranking_strategy(ranking_strategy)
    else:
        strategy = "segmenter"
    if not tweets:
        return {"results": []}

    matcher = TwitterTextMatcher()
    extracted_hashtags = matcher(tweets)
    occurrence_count = sum(len(hashtags) for hashtags in extracted_hashtags)
    if occurrence_count > MAX_TWEET_HASHTAG_OCCURRENCES:
        raise ValueError(
            "tweets must contain at most "
            f"{MAX_TWEET_HASHTAG_OCCURRENCES} hashtag occurrences"
        )
    if not any(extracted_hashtags):
        return {
            "results": [
                {"input": tweet, "segmented_text": tweet, "hashtags": []}
                for tweet in tweets
            ]
        }

    hashtag_set = list(
        dict.fromkeys(
            hashtag
            for tweet_hashtags in extracted_hashtags
            for hashtag in tweet_hashtags
        )
    )
    _validate_input_count(hashtag_set, "unique tweet hashtags")
    _validate_hashtag_lengths(hashtag_set)
    normalized_hashtags = _normalize_inputs(
        hashtag_set,
        lower=word_lower,
        remove_hashtag=True,
        hashtag_character="#",
    )
    replacement_options = {
        "hashtag_token": hashtag_token,
        "lower": lower,
        "separator": separator,
        "hashtag_character": hashtag_character,
    }
    if segmenter_kind == "transformer":
        _validate_beam_work(normalized_hashtags, top_k, steps)
        word_segmenter = get_segmenter()
        with _inference_lock:
            word_output = _run_transformer_pipeline(
                word_segmenter,
                hashtag_set,
                top_k=top_k,
                steps=steps,
                alpha=alpha,
                beta=beta,
                strategy=strategy,
                preprocessing_kwargs={"lower": word_lower},
            )
    else:
        word_segmenter = _TimeoutRegexWordSegmenter(regex_rules=regex_rules)
        word_output = word_segmenter.predict(hashtag_set, lower=word_lower)

    tweet_segmenter = TweetSegmenter(matcher=matcher, word_segmenter=word_segmenter)
    container = HashtagContainer(
        extracted_hashtags,
        hashtag_set,
        tweet_segmenter.compile_dict(
            hashtag_set,
            word_output.output,
            **replacement_options,
        )
    )
    segmented_tweets = list(
        tweet_segmenter.segmented_tweet_generator(
            tweets,
            container.hashtags,
            container.hashtag_set,
            container.replacement_dict,
            flag=regex_flags,
        )
    )
    if sum(len(tweet) for tweet in segmented_tweets) > MAX_TWEET_OUTPUT_CHARS:
        raise ValueError("tweet replacements produced an oversized response")

    if segmenter_kind == "transformer":
        unique_results = _serialize_word_output(
            container.hashtag_set,
            normalized_hashtags,
            word_output,
            strategy,
            max_candidates,
            include_component_rankings,
        )
    else:
        unique_results = [
            _regex_result(original, normalized, segmentation)
            for original, normalized, segmentation in zip(
                container.hashtag_set,
                normalized_hashtags,
                word_output.output,
            )
        ]
    results_by_hashtag = {result["input"]: result for result in unique_results}
    return {
        "results": [
            {
                "input": tweet,
                "segmented_text": segmented_tweet,
                "hashtags": [results_by_hashtag[hashtag] for hashtag in hashtags],
            }
            for tweet, segmented_tweet, hashtags in zip(
                tweets,
                segmented_tweets,
                container.hashtags,
            )
        ]
    }


@mcp.tool(annotations=MODEL_READ_ANNOTATIONS)
def rank_candidates(
    candidate_sets: list[CandidateSetInput],
    ranking_strategy: RankingStrategy = "auto",
    alpha: float = 0.222,
    beta: float = 0.111,
    max_candidates: int = 5,
    include_component_rankings: bool = False,
) -> SegmentationsResult:
    """Select, rerank, or ensemble caller-supplied segmentation candidates.

    Args:
        candidate_sets: Precomputed hypotheses grouped by unsegmented input.
        ranking_strategy: Component used to select each result.
        alpha: Segmenter-score weight used by the ensemble.
        beta: Reranker-score weight used by the ensemble.
        max_candidates: Response limit per ranking, capped at 64.
        include_component_rankings: Whether to return every component rank.

    Returns:
        Ranked candidate sets without rerunning beam search.

    Raises:
        ValueError: If candidate sets, scores, or ranking options are invalid.
    """
    if not isinstance(candidate_sets, list):
        raise ValueError("candidate_sets must be a list")
    _validate_input_count(candidate_sets, "candidate_sets")
    _validate_finite(alpha, "alpha")
    _validate_finite(beta, "beta")
    _validate_max_candidates(max_candidates)
    strategy = _resolve_ranking_strategy(ranking_strategy)

    prepared = []
    total_candidates = 0
    for candidate_set in candidate_sets:
        if not isinstance(candidate_set, dict):
            raise ValueError("each candidate set must be an object")
        original_input = candidate_set.get("input")
        candidates = candidate_set.get("candidates")
        if (
            not isinstance(original_input, str)
            or not original_input.lstrip("#").strip()
        ):
            raise ValueError("candidate set inputs must contain text")
        _validate_hashtag_lengths([original_input])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("each candidate set must contain candidates")
        total_candidates += len(candidates)
        if total_candidates > MAX_PRECOMPUTED_CANDIDATES:
            raise ValueError(
                "candidate_sets must contain at most "
                f"{MAX_PRECOMPUTED_CANDIDATES} total candidates"
            )
        normalized_input = original_input.lstrip("#")
        expected_characters = normalized_input.replace(" ", "")
        scores = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("each candidate must be an object")
            segmentation = candidate.get("segmentation")
            score = candidate.get("score")
            if not isinstance(segmentation, str) or not segmentation.strip():
                raise ValueError("candidate segmentations must contain text")
            if segmentation != " ".join(segmentation.split()):
                raise ValueError(
                    "candidate segmentations must use single spaces as boundaries"
                )
            maximum_length = max(1, (2 * len(expected_characters)) - 1)
            if len(segmentation) > maximum_length:
                raise ValueError("candidate segmentation contains too many boundaries")
            if segmentation in scores:
                raise ValueError("candidate segmentations must be unique within a set")
            if segmentation.replace(" ", "") != expected_characters:
                raise ValueError(
                    "candidate segmentations must match their unsegmented input"
                )
            _validate_finite(score, "candidate score")
            scores[segmentation] = float(score)
        prepared.append(
            {
                "index": len(prepared),
                "original_input": original_input,
                "normalized_input": normalized_input,
                "characters": expected_characters,
                "scores": scores,
            }
        )
    if not prepared:
        return {"results": []}

    batches = []
    batch_characters = []
    for item in prepared:
        for batch, characters in zip(batches, batch_characters):
            if item["characters"] not in characters:
                batch.append(item)
                characters.add(item["characters"])
                break
        else:
            batches.append([item])
            batch_characters.append({item["characters"]})

    results = [None] * len(prepared)

    def process_batch(batch, segmenter=None) -> None:
        """Rank one batch whose candidate character sets do not collide.

        Args:
            batch: Prepared candidate sets with distinct underlying characters.
            segmenter: Configured Transformer wrapper for model-backed ranking.

        Returns:
            None.
        """
        combined_scores = {
            segmentation: score
            for item in batch
            for segmentation, score in item["scores"].items()
        }
        combined_run = ProbabilityDictionary(combined_scores)
        original_inputs = [item["original_input"] for item in batch]
        normalized_inputs = [item["normalized_input"] for item in batch]
        if strategy == "segmenter":
            output = WordSegmenterOutput(
                output=combined_run.get_segmentations(
                    astype="list",
                    gold_array=normalized_inputs,
                ),
                segmenter_rank=combined_run.to_dataframe(),
            )
        else:
            output = _run_transformer_pipeline(
                segmenter,
                normalized_inputs,
                top_k=5,
                steps=5,
                alpha=alpha,
                beta=beta,
                strategy=strategy,
                segmenter_run=combined_run,
                preprocessing_kwargs={"remove_hashtag": False},
            )
        serialized = _serialize_word_output(
            original_inputs,
            normalized_inputs,
            output,
            strategy,
            max_candidates,
            include_component_rankings,
        )
        for item, result in zip(batch, serialized):
            results[item["index"]] = result

    if strategy == "segmenter":
        for batch in batches:
            process_batch(batch)
    else:
        with _inference_lock:
            segmenter = get_segmenter()
            for batch in batches:
                process_batch(batch, segmenter)
    return {"results": results}


def main(argv: list[str] | None = None) -> None:
    """Configure and run the Hashformers MCP server over stdio.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        None.
    """
    configure_server(parse_server_config(argv))
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
