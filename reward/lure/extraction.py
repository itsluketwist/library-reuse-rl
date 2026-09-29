"""Split an LLM response into reasoning and answer, and extract the answer's python code."""

import re
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from lure.analysis import parse_python


# the (opening, closing) tag pairs that wrap a model's reasoning by default
DEFAULT_REASONING_TAGS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
    ("<thought>", "</thought>"),
)

# reasoning tags, given as (opening, closing) pairs, or as a single pair
ReasoningTags = Iterable[tuple[str, str]] | tuple[str, str]

# a complete markdown code fence: ```<info string>\n<code>```
# an unterminated fence (e.g. a response cut off by the token limit) never matches
_FENCE = re.compile(
    r"```[ \t]*(?P<info>[^\n`]*)\r?\n(?P<code>.*?)```",
    re.DOTALL,
)

# fence labels that mark a block as python
PYTHON_LANGUAGES: frozenset[str] = frozenset({"py", "python", "python3"})


def normalise_reasoning_tags(
    reasoning_tags: ReasoningTags,
) -> tuple[tuple[str, str], ...]:
    """
    Check reasoning tags, accepting either (opening, closing) pairs or a single pair.

    Returns the tags as a tuple of (opening, closing) pairs.
    """
    # a single pair, e.g. ("[THINK]", "[/THINK]"), rather than a collection of pairs
    is_single_pair = (
        isinstance(reasoning_tags, tuple)
        and len(reasoning_tags) == 2
        and all(isinstance(tag, str) for tag in reasoning_tags)
    )
    # typed as Any, since users may pass anything here and every item is checked below
    candidates: Iterable[Any] = [reasoning_tags] if is_single_pair else reasoning_tags

    pairs = []
    for pair in candidates:
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or not all(isinstance(tag, str) and tag for tag in pair)
        ):
            raise ValueError(
                f"Reasoning tags must be (opening, closing) pairs of strings, got {pair!r}.",
            )
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


@lru_cache(maxsize=32)
def _reasoning_patterns(
    reasoning_tags: tuple[tuple[str, str], ...],
) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """
    Build the regexes that find the reasoning tags, ignoring case. They are cached, since
    the same tags are used for every response.

    Returns an (opening, closing) pair of patterns. The opening pattern only matches at the
    start of the response.
    """
    openings = "|".join(re.escape(opening) for opening, _ in reasoning_tags)
    closings = "|".join(re.escape(closing) for _, closing in reasoning_tags)
    return (
        re.compile(rf"^\s*(?:{openings})\s*", re.IGNORECASE),
        re.compile(closings, re.IGNORECASE),
    )


def _is_valid_python(
    code: str,
) -> bool:
    """
    Check whether some text parses as python.

    Returns True when the text is valid python.
    """
    try:
        parse_python(code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    return True


def split_reasoning(
    response: str,
    reasoning_tags: ReasoningTags = DEFAULT_REASONING_TAGS,
    require_reasoning: bool = False,
) -> tuple[str, str]:
    """
    Split a response into its reasoning and its final answer.

    Everything before the first closing reasoning tag (e.g. </think>) is reasoning, and
    everything after it is the answer. Only the closing tag is required, because some chat
    templates (e.g. Qwen3.5) put the opening <think> tag in the prompt rather than the
    response. If there is an opening tag but no closing tag, the model never finished
    reasoning, so there is no answer.

    A response with no tags at all is ambiguous. By default it is all answer (e.g. from a
    model that doesn't reason). With `require_reasoning`, the model is known to reason
    first, so it is all unfinished reasoning (e.g. cut off by the token limit before
    </think>, when the template put <think> in the prompt), and there is no answer.

    `reasoning_tags` are the (opening, closing) pairs to look for, matched ignoring case,
    e.g. [("[THINK]", "[/THINK]")]. By default the common xml-style tags are used (<think>,
    <thinking>, <reasoning> and <thought>). With no tags, the whole response is the answer.

    Returns a (reasoning, answer) tuple, both stripped of surrounding whitespace.
    """
    tags = normalise_reasoning_tags(reasoning_tags)
    if not tags:
        return "", response.strip()
    opening, closing = _reasoning_patterns(tags)

    close_match = closing.search(response)
    if close_match is not None:
        reasoning = opening.sub("", response[: close_match.start()])
        return reasoning.strip(), response[close_match.end() :].strip()

    # an opening tag with no closing tag: still reasoning when the response ended
    if opening.search(response):
        return opening.sub("", response).strip(), ""

    # no tags at all: unfinished reasoning if the model must reason first, else the answer
    if require_reasoning:
        return response.strip(), ""
    return "", response.strip()


def extract_code_blocks(
    answer: str,
) -> list[str]:
    """
    Extract the python code from an answer (the response with reasoning removed).

    Complete fenced code blocks labelled as python are extracted in the order they appear,
    even if their code is broken. Unlabelled fences often hold other things (shell commands,
    program output, directory trees), so one only counts as python when its code parses.
    Fences in other languages (e.g. bash) are skipped. If the answer has no fences at all,
    but the whole answer is valid python, it is treated as one raw code block. Prose is
    never treated as code, and neither is an unterminated fence.

    Returns the python code blocks, each stripped of surrounding whitespace.
    """
    matches = list(_FENCE.finditer(answer))
    if matches:
        blocks = []
        for match in matches:
            # the language is the first word of the info string, e.g. "python title=x"
            info = match.group("info").strip().lower().split()
            language = info[0] if info else ""
            code = match.group("code").strip()
            if not code:
                continue
            if language in PYTHON_LANGUAGES or (
                not language and _is_valid_python(code)
            ):
                blocks.append(code)
        return blocks

    # a lone ``` is a truncated or malformed fence, not a raw python answer
    if "```" in answer or not answer.strip():
        return []

    # fall back to the whole answer, but only when it is entirely valid python
    return [answer.strip()] if _is_valid_python(answer) else []
