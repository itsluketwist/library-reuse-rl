"""Tested-implementation reward: execute the generated code against the record's tests."""

from typing import Any

from src.grpo.rewards.base import _IO_BOILERPLATE_LIB, CodeReward
from src.response.parser import ParsedResponse
from src.response.pipeline import process_response


def execution_components(
    parsed: ParsedResponse,
    passed: bool,
) -> dict[str, bool]:
    """
    Judge the execution-specific components of the tested-implementation reward.

    correct_format and passes_syntax_check are shared with the other rewards (see
    LibraryReward.has_correct_format / CodeReward.has_valid_syntax) and judged by
    TestedImplementationReward.components, not here.

    Returns each component name mapped to whether the completion satisfies it.
    """
    # stdlib imports, excluding the io-boilerplate exception (see base.py)
    internal_libs = [lib for lib in parsed.std_libs if lib != _IO_BOILERPLATE_LIB]

    return {
        # the code passed evaluation against the record's tests
        "passed_tests": passed,
        # imports a stdlib module (other than the io boilerplate exception)
        "internal_lib_imported": bool(internal_libs),
        # imports a third-party module
        "external_lib_imported": bool(parsed.ext_libs),
        # actually calls/uses an imported stdlib module
        "internal_lib_used": any(lib in parsed.lib_usage for lib in internal_libs),
        # actually calls/uses an imported third-party module
        "external_lib_used": any(lib in parsed.lib_usage for lib in parsed.ext_libs),
    }


def compute_reward(
    parsed: ParsedResponse,
    passed: bool,
    weights: dict[str, float],
) -> float:
    """
    Compute the execution-specific part of the tested-implementation reward.

    Returns a scalar reward.
    """
    components = execution_components(parsed=parsed, passed=passed)
    return sum(
        (weights.get(component, 0.0) for component, ok in components.items() if ok),
        0.0,
    )


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
            "correct_format": self.has_correct_format(processed),
            "passes_syntax_check": self.has_valid_syntax(processed),
            **execution_components(parsed=processed, passed=processed.passed),
        }
