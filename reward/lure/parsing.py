"""Parse a whole LLM response: reasoning, answer, code blocks, and library usage."""

from collections.abc import Iterable
from dataclasses import dataclass

from lure.analysis import CodeBlock, analyse_code
from lure.extraction import (
    DEFAULT_REASONING_TAGS,
    ReasoningTags,
    extract_code_blocks,
    split_reasoning,
)


@dataclass(frozen=True)
class ParsedResponse:
    """
    Everything lure finds in one LLM response.

    The library sets combine every python code block in the answer. A library counts as
    used only when a single block both imports it and references it, so an import in one
    example and a call in another never add up to fake usage. Reasoning is never analysed,
    so thinking about a library earns nothing.
    """

    reasoning: str
    answer: str
    stop_reason: str | None
    code_blocks: tuple[CodeBlock, ...]

    @property
    def truncated(
        self,
    ) -> bool:
        """
        Whether generation stopped because it hit the token limit.

        Returns True when the stop reason is "length" (the vLLM / OpenAI finish reason).
        """
        return self.stop_reason == "length"

    @property
    def valid(
        self,
    ) -> bool:
        """
        Whether the answer contains python code that is all syntactically valid.

        Returns True when there is at least one code block and every block parses.
        """
        return bool(self.code_blocks) and all(block.valid for block in self.code_blocks)

    @property
    def external_imports(
        self,
    ) -> frozenset[str]:
        """
        The external (third-party) libraries imported by any code block.

        Returns the top-level import names, e.g. {"PIL", "requests"}.
        """
        return frozenset().union(
            *(block.external_imports for block in self.code_blocks)
        )

    @property
    def external_used(
        self,
    ) -> frozenset[str]:
        """
        The external libraries that are imported and then used in the same code block.

        Returns the top-level import names.
        """
        return frozenset().union(*(block.external_used for block in self.code_blocks))

    @property
    def stdlib_imports(
        self,
    ) -> frozenset[str]:
        """
        The standard-library modules imported by any code block.

        Returns the top-level module names.
        """
        return frozenset().union(*(block.stdlib_imports for block in self.code_blocks))

    @property
    def stdlib_used(
        self,
    ) -> frozenset[str]:
        """
        The standard-library modules that are imported and then used in the same block.

        Returns the top-level module names.
        """
        return frozenset().union(*(block.stdlib_used for block in self.code_blocks))

    @property
    def longest_block(
        self,
    ) -> CodeBlock | None:
        """
        The longest code block by line count (the first one wins a tie), for code that
        needs to pick a single program to run.

        Returns the longest CodeBlock, or None when there is no code.
        """
        if not self.code_blocks:
            return None
        return max(self.code_blocks, key=lambda block: len(block.code.splitlines()))


def parse_response(
    response: str | None,
    stop_reason: str | None = None,
    local_modules: Iterable[str] = (),
    reasoning_tags: ReasoningTags = DEFAULT_REASONING_TAGS,
    require_reasoning: bool = False,
) -> ParsedResponse:
    """
    Parse an LLM response into its reasoning, answer, and analysed python code blocks.

    `stop_reason` is how generation ended (e.g. vLLM's `finish_reason`: "stop" or
    "length"), used to detect truncated responses. `local_modules` are the top-level names
    of the project's own modules, which are never counted as external libraries.
    `reasoning_tags` are the (opening, closing) pairs that wrap the model's reasoning, e.g.
    [("[THINK]", "[/THINK]")]. Set `require_reasoning` for a model that always reasons
    first, so a response with no tags counts as unfinished reasoning rather than an answer -
    see split_reasoning.

    Returns the ParsedResponse.
    """
    reasoning, answer = split_reasoning(
        response or "",
        reasoning_tags=reasoning_tags,
        require_reasoning=require_reasoning,
    )
    local = list(local_modules)
    code_blocks = tuple(
        analyse_code(code, local_modules=local) for code in extract_code_blocks(answer)
    )
    return ParsedResponse(
        reasoning=reasoning,
        answer=answer,
        stop_reason=stop_reason,
        code_blocks=code_blocks,
    )
