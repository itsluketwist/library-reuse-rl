"""Parse LLM responses into reasoning, answer text, and analysed code blocks."""

import re
from dataclasses import dataclass

from llm_cgr import CodeBlock, Markdown
from radon.metrics import mi_visit
from radon.raw import analyze


# opening reasoning tag at the start of a string, stripped from the reasoning block
_REASONING_OPEN_TAG = re.compile(
    r"^\s*<(?:think|thinking|reasoning|thought)>\s*",
    re.IGNORECASE,
)
_REASONING_CLOSE_TAG = re.compile(
    r"</(?:think|thinking|reasoning|thought)>",
    re.IGNORECASE,
)


@dataclass
class ParsedCodeBlock:
    """One extracted code block, including analysis and test-candidate status."""

    language: str | None
    text: str
    valid: bool | None
    error: str | None
    std_libs: list[str]
    ext_libs: list[str]
    lib_usage: dict[str, list[dict]]
    logical_lines: int | None
    maintainability_index: float | None
    selected_for_tests: bool


@dataclass
class ParsedResponse:
    """A response split into reasoning, answer, and all analysed answer code blocks."""

    reasoning: str
    answer: str | None  # None if no answer was extracted (e.g. reasoning overflow)
    code_blocks: list[ParsedCodeBlock]

    @property
    def test_code_block(self) -> ParsedCodeBlock | None:
        """The largest code block selected as the executable test candidate.

        Returns the marked block, or None when the answer contains no code block.
        """
        return next(
            (block for block in self.code_blocks if block.selected_for_tests),
            None,
        )

    @property
    def code(self) -> str | None:
        """Candidate source code retained for tested-task compatibility.

        Returns the selected test block's text, or None.
        """
        block = self.test_code_block
        return block.text if block is not None else None

    @property
    def valid(self) -> bool | None:
        """Candidate syntax validity retained for tested-task compatibility.

        Returns the selected test block's validity, or None.
        """
        block = self.test_code_block
        return block.valid if block is not None else None

    @property
    def error(self) -> str | None:
        """Candidate parse error retained for tested-task compatibility.

        Returns the selected test block's error, or None.
        """
        block = self.test_code_block
        return block.error if block is not None else None

    @property
    def std_libs(self) -> list[str]:
        """Candidate standard-library imports retained for tested-task compatibility.

        Returns the selected test block's standard-library imports.
        """
        block = self.test_code_block
        return block.std_libs if block is not None else []

    @property
    def ext_libs(self) -> list[str]:
        """Candidate external imports retained for tested-task compatibility.

        Returns the selected test block's external imports.
        """
        block = self.test_code_block
        return block.ext_libs if block is not None else []

    @property
    def lib_usage(self) -> dict[str, list[dict]]:
        """Candidate library usage retained for tested-task compatibility.

        Returns the selected test block's usage mapping.
        """
        block = self.test_code_block
        return block.lib_usage if block is not None else {}

    @property
    def logical_lines(self) -> int | None:
        """Candidate logical-line count retained for tested-task compatibility.

        Returns the selected test block's logical-line count, or None.
        """
        block = self.test_code_block
        return block.logical_lines if block is not None else None

    @property
    def maintainability_index(self) -> float | None:
        """Candidate maintainability retained for tested-task compatibility.

        Returns the selected test block's maintainability index, or None.
        """
        block = self.test_code_block
        return block.maintainability_index if block is not None else None


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
    has no fenced code blocks, an empty fragment is returned - we only analyse code
    the model explicitly presented as code.

    Returns the selected CodeBlock, already analysed for library usage.
    """
    code_blocks = list(Markdown(text=answer).code_blocks)
    selected_index = _test_code_block_index(code_blocks)
    if selected_index is not None:
        return code_blocks[selected_index]

    return CodeBlock(language="python", text="")


def _test_code_block_index(
    code_blocks: list[CodeBlock],
) -> int | None:
    """Find the largest block that test-backed tasks should execute.

    Returns the block index, or None when there are no blocks.
    """
    if not code_blocks:
        return None
    return max(
        range(len(code_blocks)),
        key=lambda index: len(code_blocks[index].text.splitlines()),
    )


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
    Parse an LLM response into its reasoning, answer, and analysed code blocks.

    Every fenced answer block is retained and analysed. The longest block is marked as
    the candidate program for test-backed tasks; non-executing analyses can inspect all
    blocks without parsing the response again.

    Returns the complete parsed response.
    """
    reasoning, answer = strip_reasoning(response)
    source_blocks = list(Markdown(text=answer).code_blocks)
    selected_index = _test_code_block_index(source_blocks)

    code_blocks = []
    for index, block in enumerate(source_blocks):
        logical_lines, maintainability_index = code_metrics(block.text)
        code_blocks.append(
            ParsedCodeBlock(
                language=block.language,
                text=block.text,
                valid=block.valid,
                error=block.error,
                std_libs=block.std_libs,
                ext_libs=block.ext_libs,
                lib_usage=block.lib_usage,
                logical_lines=logical_lines,
                maintainability_index=maintainability_index,
                selected_for_tests=index == selected_index,
            )
        )

    return ParsedResponse(
        reasoning=reasoning,
        # store None rather than empty strings when nothing was extracted
        answer=answer or None,
        code_blocks=code_blocks,
    )
