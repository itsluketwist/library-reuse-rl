"""Parse LLM responses to find the candidate code fragment and analyse it."""

import re
from dataclasses import dataclass

from llm_cgr import CodeBlock, Markdown
from radon.metrics import mi_visit
from radon.raw import analyze


# matches the opening reasoning tag at the very start of a string (with optional
# surrounding whitespace), used to strip it from the extracted reasoning block
_REASONING_OPEN_TAG = re.compile(
    r"^\s*<(?:think|thinking|reasoning|thought)>\s*",
    re.IGNORECASE,
)
_REASONING_CLOSE_TAG = re.compile(
    r"</(?:think|thinking|reasoning|thought)>",
    re.IGNORECASE,
)


@dataclass
class ParsedResponse:
    """
    A response, split into reasoning and answer, with the candidate code analysed.
    """

    reasoning: str
    answer: str
    code: str  # the selected code fragment, may be empty if the answer had no code
    valid: bool | None  # whether the code fragment parsed as valid python
    error: str | None  # parse error message, only set if not valid
    std_libs: list[str]
    ext_libs: list[str]
    lib_usage: dict[str, list[dict]]
    logical_lines: int | None  # None if the fragment failed to parse
    maintainability_index: float | None  # None if the fragment failed to parse


def strip_reasoning(
    response: str,
) -> tuple[str, str]:
    """
    Split a response into its reasoning preamble and answer.

    Looks for the first closing reasoning tag (e.g. </think>) and treats everything
    before it as reasoning, everything after as the answer. The opening tag (e.g.
    <think>) is stripped from the start of the reasoning so it contains only the
    model's thoughts. If no closing tag is found, the whole response is the answer
    (empty reasoning).

    Returns a (reasoning, answer) tuple.
    """
    match = _REASONING_CLOSE_TAG.search(response)
    if match:
        reasoning = _REASONING_OPEN_TAG.sub("", response[: match.start()])
        return reasoning, response[match.end() :].strip()

    # open tag with no close tag means the model is still reasoning - no answer
    if _REASONING_OPEN_TAG.search(response):
        return _REASONING_OPEN_TAG.sub("", response), ""

    return "", response


def select_code(
    answer: str,
) -> CodeBlock:
    """
    Select the code fragment to analyse from an answer.

    Picks the longest (by line count) fenced code block in the answer. If the answer
    has no fenced code blocks, the whole answer is treated as a single python
    fragment, since models sometimes respond with bare code and no markdown fences.

    Returns the selected CodeBlock, already analysed for library usage.
    """
    code_blocks = Markdown(text=answer).code_blocks
    if code_blocks:
        return max(code_blocks, key=lambda block: len(block.text.splitlines()))

    return CodeBlock(language="python", text="")


def code_metrics(
    code: str,
) -> tuple[int | None, float | None]:
    """
    Compute the logical lines of code and maintainability index for a code fragment.

    The fragment came from an LLM response rather than trusted source, so any parse
    failure (radon parses with both ast and tokenize internally) is caught and
    reported as missing metrics rather than raising.

    Returns a (logical_lines, maintainability_index) tuple, with None values on parse failure.
    """
    try:
        logical_lines = analyze(code).lloc
        maintainability_index = mi_visit(code, multi=True)
    except Exception:
        return None, None

    return logical_lines, maintainability_index


def parse_response(
    response: str,
) -> ParsedResponse:
    """
    Parse an LLM response into its reasoning, answer, and analysed code fragment.

    Returns a ParsedResponse with the selected code fragment's library usage and
    code metrics.
    """
    reasoning, answer = strip_reasoning(response)
    block = select_code(answer)
    logical_lines, maintainability_index = code_metrics(block.text)

    return ParsedResponse(
        reasoning=reasoning,
        answer=answer,
        code=block.text,
        # blank code means no code was produced - treat as missing rather than valid
        valid=None if not block.text else block.valid,
        error=block.error,
        std_libs=block.std_libs,
        ext_libs=block.ext_libs,
        lib_usage=block.lib_usage,
        logical_lines=logical_lines,
        maintainability_index=maintainability_index,
    )
