"""The individual toolbox commands - small, focused helpers the team reuses.

New commands go here; keep each one small and register it in cli.py. These are intentionally
minimal stubs - real behaviour is filled in as the helpers are needed.
"""

from pathlib import Path

from toolbox.utils import read_text, write_text


def count_lines(
    path: str | Path,
) -> int:
    """Count the lines in a text file.

    Returns the line count.
    """
    return len(read_text(path).splitlines())


def slugify_title(
    title: str,
) -> str:
    """Turn a title into a rough url-safe slug (placeholder - improve as needed).

    Returns the slug.
    """
    return "-".join(title.lower().split())


def summarise_file(
    path: str | Path,
    out_path: str | Path,
) -> None:
    """Write a tiny summary (line count) of a file to another file."""
    summary = f"{Path(path).name}: {count_lines(path)} lines\n"
    write_text(out_path, summary)
