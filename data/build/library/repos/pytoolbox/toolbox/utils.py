"""Shared helpers used across the toolbox commands."""

from pathlib import Path


def read_text(
    path: str | Path,
) -> str:
    """Read a text file's contents.

    Returns the file text.
    """
    return Path(path).read_text(encoding="utf-8")


def write_text(
    path: str | Path,
    content: str,
) -> None:
    """Write text to a file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def print_rows(
    rows: list[dict],
) -> None:
    """Print a list of record dicts, one per line, for quick inspection."""
    for row in rows:
        print(row)
