# ml4_mini_swe_LRB_T003

````python
"""Text similarity comparison utilities.

Provides :func:`text_similarity_scores`, which compares two texts and
returns a ``(cosine, ratio)`` tuple of floats.
"""

import math
import re
from collections import Counter

try:  # Fast C implementation of the indel (insertion/deletion) ratio.
    from Levenshtein import ratio as _levenshtein_ratio
except Exception:  # pragma: no cover - fallback when package is missing
    _levenshtein_ratio = None


# Every run of characters that are not letters or digits (underscores
# count as "other" characters) is replaced with a single space.
_NON_ALNUM_RUN = re.compile(r"[\W_]+", re.UNICODE)


def _clean(text):
    """Lowercase ``text`` and collapse non-alphanumeric runs to one space."""
    if text is None:
        text = ""
    return _NON_ALNUM_RUN.sub(" ", str(text).lower())


def _lcs_length(a, b):
    """Length of the longest common subsequence of ``a`` and ``b``."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return 0
    previous = [0] * (len(b) + 1)
    for char_a in a:
        current = [0] * (len(b) + 1)
        for j, char_b in enumerate(b, 1):
            if char_a == char_b:
                current[j] = previous[j - 1] + 1
            else:
                up = previous[j]
                left = current[j - 1]
                current[j] = up if up >= left else left
        previous = current
    return previous[len(b)]


def _indel_ratio(a, b):
    """0-1 similarity based on insertions/deletions turning ``a`` into ``b``.

    The minimal number of insertions and deletions needed is
    ``len(a) + len(b) - 2 * LCS(a, b)``; the ratio is
    ``1 - (insertions + deletions) / (len(a) + len(b))`` and is 1.0 when
    both strings are empty.  Matches ``Levenshtein.ratio`` bit for bit.
    """
    if _levenshtein_ratio is not None:
        return float(_levenshtein_ratio(a, b))
    total = len(a) + len(b)
    if total == 0:
        return 1.0
    edits = total - 2 * _lcs_length(a, b)
    return 1.0 - edits / total


def _cosine_similarity(a, b):
    """Cosine similarity of the word-count vectors of ``a`` and ``b``."""
    words_a = a.split()
    words_b = b.split()
    if not words_a or not words_b:
        return 0.0
    counts_a = Counter(words_a)
    counts_b = Counter(words_b)
    dot = sum(count * counts_b[word] for word, count in counts_a.items())
    norm_a = math.sqrt(sum(count * count for count in counts_a.values()))
    norm_b = math.sqrt(sum(count * count for count in counts_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot) / (norm_a * norm_b)


def text_similarity_scores(text1, text2):
    """Compare two texts and return ``(cosine, ratio)`` as floats.

    Both texts are lowercased and every run of characters other than
    letters and digits (underscores included) is replaced with a single
    space.  ``cosine`` is the cosine similarity of the two cleaned texts'
    word counts (0.0 if either has no words).  ``ratio`` is a 0-1
    similarity based on the insertions and deletions needed to turn one
    cleaned string into the other (1.0 if both are empty).
    """
    cleaned1 = _clean(text1)
    cleaned2 = _clean(text2)
    cosine = _cosine_similarity(cleaned1, cleaned2)
    ratio = _indel_ratio(cleaned1, cleaned2)
    return (float(cosine), float(ratio))
````
