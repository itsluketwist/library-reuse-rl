"""Shared scoring helpers: rates, averages, and library-use statistics."""

from collections import Counter
from typing import Any


def _rate(
    count: int,
    total: int,
) -> float:
    """Fraction count/total, or 0.0 when there is nothing to divide.

    Returns the rate.
    """
    return count / total if total else 0.0


def _average(
    values: list[float],
) -> float | None:
    """Average a list of numbers.

    Returns the mean, or None if there are no values to average.
    """
    return sum(values) / len(values) if values else None


def _library_statistics(
    library_sets: list[set[str]],
) -> dict[str, Any]:
    """Summarise distinct external libraries used by each response.

    Returns reuse rate, average library count, and per-library response counts.
    """
    counts: Counter[str] = Counter()
    for libraries in library_sets:
        counts.update(libraries)

    return {
        "using_lib_rate": _rate(
            sum(bool(libs) for libs in library_sets),
            len(library_sets),
        ),
        "avg_extlibs": _average([len(libs) for libs in library_sets]),
        "library_counts": dict(counts.most_common()),
    }
