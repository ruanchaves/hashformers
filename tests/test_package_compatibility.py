import os
import runpy
import subprocess
import sys
from pathlib import Path

import setuptools

import hashformers


ROOT = Path(__file__).parent.parent


def _source_environment():
    """Return an environment that imports this checkout instead of an old wheel."""
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_path}{os.pathsep}{existing_path}"
        if existing_path
        else source_path
    )
    return environment


def read_setup_kwargs(monkeypatch):
    """Execute setup.py while recording its package metadata.

    Args:
        monkeypatch: Pytest fixture used to replace ``setuptools.setup``.

    Returns:
        The keyword arguments passed to ``setuptools.setup``.
    """
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    monkeypatch.chdir(ROOT)
    runpy.run_path(str(ROOT / "setup.py"))
    return captured


def test_supported_python_and_transformers_versions(monkeypatch):
    metadata = read_setup_kwargs(monkeypatch)

    assert metadata["version"] == "3.0.0"
    assert metadata["python_requires"] == ">=3.10"
    assert "transformers>=4.46.1,<6" in metadata["install_requires"]
    assert "Programming Language :: Python :: 3.8" not in metadata["classifiers"]
    assert "Programming Language :: Python :: 3.9" not in metadata["classifiers"]


def test_legacy_regex_and_tweet_apis_are_not_exported():
    for name in (
        "RegexWordSegmenter",
        "TweetSegmenter",
        "TwitterTextMatcher",
        "TweetSegmenterOutput",
        "HashtagContainer",
    ):
        assert not hasattr(hashformers, name)


def test_package_import_does_not_load_model_runtime():
    """Keep lightweight submodules usable without importing the ML stack."""
    code = """
import sys
import hashformers

blocked = {"minicons", "pandas", "torch", "transformers"}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"eagerly loaded model dependencies: {loaded}")
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_source_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_documented_transformer_export_resolves_lazily():
    """Preserve the documented top-level model import after lazy loading."""
    from hashformers import TransformerWordSegmenter
    from hashformers.segmenter.auto import (
        TransformerWordSegmenter as DirectTransformerWordSegmenter,
    )

    assert TransformerWordSegmenter is DirectTransformerWordSegmenter


def test_lazy_package_preserves_historical_public_names():
    """Keep the pre-lazy top-level import surface discoverable."""
    assert set(hashformers.__all__) == {
        "Any",
        "BaseSegmenter",
        "BaseWordSegmenter",
        "Beamsearch",
        "DEFAULT_MAX_BATCH_SIZE",
        "ReciprocalRankFusionEnsembler",
        "Reranker",
        "Top2_Ensembler",
        "TransformerWordSegmenter",
        "WordSegmenterOutput",
        "base_segmenter",
        "beamsearch",
        "data_structures",
        "enforce_prob_dict",
        "ensemble",
        "evaluation",
        "experiments",
        "segmenter",
    }
    assert set(hashformers.__all__).issubset(dir(hashformers))


def test_mcp_server_is_an_optional_extra(monkeypatch):
    metadata = read_setup_kwargs(monkeypatch)

    assert metadata["extras_require"]["mcp"] == [
        "anyio>=4.9",
        "huggingface-hub>=0.26,<2",
        "mcp>=2,<3",
    ]
    assert all(
        not dependency.startswith("mcp")
        for dependency in metadata["install_requires"]
    )
    assert metadata["entry_points"]["console_scripts"] == [
        "hashformers-mcp=hashformers.mcp_server:main",
    ]


def test_dependency_files_and_readme_match_package_metadata(monkeypatch):
    metadata = read_setup_kwargs(monkeypatch)
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "transformers>=4.46.1,<6" in requirements
    assert "twitter-text-python" not in requirements
    assert "twitter-text-python" not in metadata["install_requires"]
    assert "Python 3.10 or newer" in readme
    assert "Transformers 4.46.1" in readme


def test_readme_documents_mcp_and_agent_skill_setup():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert 'pip install "hashformers[mcp]"' in readme
    assert "codex mcp add hashformers -- hashformers-mcp" in readme
    assert "claude mcp add --transport stdio --scope user" in readme
    assert ".agents/skills/segment-hashtags" in readme
    assert "~/.claude/skills" in readme
    assert "--defer-model-selection" in readme
    assert "sample_hashtag_file" in readme
    assert "discover_huggingface_models" in readme
    assert "configure_models" in readme
