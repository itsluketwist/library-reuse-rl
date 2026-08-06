"""Deterministically extract which tool a prose recommendation picks, for the reward."""

import re
from dataclasses import dataclass, field


# how many characters before a match to scan for a negation cue ("don't use X")
_NEGATION_WINDOW = 30

# phrases that flip a nearby tool mention from an endorsement into a rejection
_NEGATION_CUES = (
    "don't",
    "do not",
    "dont",
    "avoid",
    "instead of",
    "rather than",
    "not use",
    "no need for",
    "without",
    "skip",
)


@dataclass
class Pick:
    """
    The outcome of reading a recommendation: what it picked and whether that reuses code.
    """

    primary: str | None  # canonical name of the primary pick, or None if none found
    is_external: bool  # the primary pick is an external library/service (reuse)
    is_diy: bool  # the primary pick is a roll-your-own / custom choice
    mentioned_external: list[str] = field(
        default_factory=list
    )  # all external names seen


def _term_pattern(
    term: str,
) -> re.Pattern[str]:
    """
    Build a case-insensitive, boundary-aware pattern for a tool name or phrase.

    The term is matched literally (so dots in names like "next.js" or spaces in phrases
    like "json file" are fine), but only on alphanumeric boundaries so "pg" doesn't match
    inside "postgres".

    Returns the compiled pattern.
    """
    return re.compile(
        rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])",
        re.IGNORECASE,
    )


def _first_position(
    text: str,
    terms: list[str],
) -> int | None:
    """
    Find the earliest position in `text` where any of `terms` appears, un-negated.

    A match is ignored when a negation cue (e.g. "don't", "instead of") appears in the
    short window just before it, so "don't use redis" is not read as picking redis.

    Returns the earliest match start, or None if no un-negated term matches.
    """
    best: int | None = None
    for term in terms:
        for match in _term_pattern(term).finditer(text):
            start = match.start()
            window = text[max(0, start - _NEGATION_WINDOW) : start]
            if any(cue in window for cue in _NEGATION_CUES):
                continue
            if best is None or start < best:
                best = start
            break  # earliest occurrence of this term is enough
    return best


def extract_primary_pick(
    response: str,
    allowlist: list[str],
    aliases: dict[str, list[str]] | None = None,
    diy_terms: list[str] | None = None,
) -> Pick:
    """
    Read a prose recommendation and decide which tool it primarily picks.

    Each allowlist tool is searched for by its canonical name and any `aliases`; the
    earliest un-negated mention across all tools and `diy_terms` is taken as the primary
    pick (recommendations lead with their main suggestion). Matching is deterministic -
    no model call - so it can run inside the training loop.

    Returns a Pick describing the primary choice and any external libraries mentioned.
    """
    aliases = aliases or {}
    diy_terms = diy_terms or []

    text = response.lower()

    # earliest position of each allowlist tool (by canonical name or any alias)
    tool_positions: dict[str, int] = {}
    for tool in allowlist:
        terms = [tool, *aliases.get(tool, [])]
        position = _first_position(text, terms)
        if position is not None:
            tool_positions[tool] = position

    mentioned_external = sorted(tool_positions, key=lambda t: tool_positions[t])

    diy_position = _first_position(text, diy_terms)

    # nothing recognised at all
    if not tool_positions and diy_position is None:
        return Pick(primary=None, is_external=False, is_diy=False)

    # the primary pick is whichever comes first - an external tool or a diy choice
    first_tool = mentioned_external[0] if mentioned_external else None
    tool_position = tool_positions.get(first_tool) if first_tool else None

    diy_first = diy_position is not None and (
        tool_position is None or diy_position < tool_position
    )
    if diy_first:
        return Pick(
            primary="custom/diy",
            is_external=False,
            is_diy=True,
            mentioned_external=mentioned_external,
        )

    return Pick(
        primary=first_tool,
        is_external=True,
        is_diy=False,
        mentioned_external=mentioned_external,
    )
