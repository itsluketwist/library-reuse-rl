"""Tested-implementation reward: execute the generated code against the record's tests."""

from typing import Any

from lure import CodeBlock

from src.grpo.rewards.base import _IO_BOILERPLATE_LIB, CodeReward
from src.response.pipeline import process_response


def execution_components(
    block: CodeBlock | None,
    passed: bool,
) -> dict[str, bool]:
    """
    Judge the execution-specific components of the tested-implementation reward, from
    the code block that was run (the answer's longest code block).

    correct_format and passes_syntax_check are shared with the other rewards (see
    LibraryReward.has_correct_format / CodeReward.has_valid_syntax) and judged by
    TestedImplementationReward.components, not here.

    Returns each component name mapped to whether the completion satisfies it.
    """
    if block is None:
        internal_imported: frozenset[str] = frozenset()
        internal_used: frozenset[str] = frozenset()
        external_imported: frozenset[str] = frozenset()
        external_used: frozenset[str] = frozenset()
    else:
        # stdlib modules, excluding the io-boilerplate exception (see base.py)
        internal_imported = block.stdlib_imports - {_IO_BOILERPLATE_LIB}
        internal_used = block.stdlib_used - {_IO_BOILERPLATE_LIB}
        external_imported = block.external_imports
        external_used = block.external_used

    return {
        # the code passed evaluation against the record's tests
        "passed_tests": passed,
        # imports a stdlib module (other than the io boilerplate exception)
        "internal_lib_imported": bool(internal_imported),
        # imports a third-party module
        "external_lib_imported": bool(external_imported),
        # actually calls/uses an imported stdlib module
        "internal_lib_used": bool(internal_used),
        # actually calls/uses an imported third-party module
        "external_lib_used": bool(external_used),
    }


class TestedImplementationReward(CodeReward):
    """
    Reward for code verified by executing it against stdin/stdout test cases.

    Runs the completion's code against the row's `inputs`/`expected_outputs` (the
    execution "pain point" this project otherwise avoids). This is the reward used for
    the existing Skywork dataset.
    """

    # the class name starts with "Test", so tell pytest it is not a test class
    __test__ = False

    def components(
        self,
        content: str,
        row: dict[str, Any],
    ) -> dict[str, bool]:
        """
        Judge one completion by running its code against the row's tests.

        Returns each component name mapped to whether the completion satisfies it.
        """
        processed = process_response(
            response=content,
            record={
                "inputs": row["inputs"],
                "expected_outputs": row["expected_outputs"],
            },
            stop_reason=self._stop_reason(row),
        )
        return {
            "correct_format": self.has_correct_format(processed.parsed),
            "passes_syntax_check": self.has_valid_syntax(processed.parsed),
            **execution_components(
                block=processed.executed_block,
                passed=processed.passed,
            ),
        }
