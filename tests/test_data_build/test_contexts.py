"""Tests for turning a repo into prompt context - here, which module names count as local."""

from pathlib import Path

from data.build.contexts import _local_modules


def _touch(
    root: Path,
    *relative_paths: str,
) -> None:
    """Create empty files at `relative_paths` under `root`, with their folders."""
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_every_package_and_module_is_local(tmp_path: Path):
    """Nested modules count too, so a bare `from utils import ...` is the project's own code."""
    _touch(
        tmp_path,
        "toolbox/__init__.py",
        "toolbox/utils.py",
        "toolbox/sub/deep.py",
        "top.py",
        "README.md",
    )

    assert _local_modules(tmp_path) == ["deep", "sub", "toolbox", "top", "utils"]


def test_caches_and_hidden_folders_are_skipped(tmp_path: Path):
    """Compiled caches and hidden folders (e.g. a virtualenv) aren't the project's modules."""
    _touch(
        tmp_path,
        "app/__init__.py",
        "app/__pycache__/cached.py",
        ".venv/lib/site.py",
        "app/__main__.py",
    )

    assert _local_modules(tmp_path) == ["app"]
