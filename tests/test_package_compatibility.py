import runpy
from pathlib import Path

import setuptools


ROOT = Path(__file__).parent.parent


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

    assert metadata["python_requires"] == ">=3.10"
    assert "transformers>=4.46.1,<6" in metadata["install_requires"]
    assert "Programming Language :: Python :: 3.8" not in metadata["classifiers"]
    assert "Programming Language :: Python :: 3.9" not in metadata["classifiers"]


def test_mcp_server_is_an_optional_extra(monkeypatch):
    metadata = read_setup_kwargs(monkeypatch)

    assert metadata["extras_require"]["mcp"] == [
        "anyio>=4.9",
        "mcp>=2,<3",
        "regex",
    ]
    assert all(
        not dependency.startswith("mcp")
        for dependency in metadata["install_requires"]
    )
    assert metadata["entry_points"]["console_scripts"] == [
        "hashformers-mcp=hashformers.mcp_server:main",
    ]


def test_dependency_files_and_readme_match_package_metadata():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "transformers>=4.46.1,<6" in requirements
    assert "Python 3.10 or newer" in readme
    assert "Transformers 4.46.1" in readme


def test_readme_documents_mcp_and_agent_skill_setup():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert 'pip install "hashformers[mcp]"' in readme
    assert "codex mcp add hashformers -- hashformers-mcp" in readme
    assert "claude mcp add --transport stdio --scope user" in readme
    assert ".agents/skills/segment-hashtags" in readme
    assert "~/.claude/skills" in readme
