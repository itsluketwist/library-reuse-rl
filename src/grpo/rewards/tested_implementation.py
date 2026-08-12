"""Tested-implementation reward: execute the generated code against the record's tests."""

from typing import Any

from src.grpo.rewards.base import _IO_BOILERPLATE_LIB, CodeReward
from src.response.parser import ParsedResponse
from src.response.pipeline import process_response


def compute_reward(
    parsed: ParsedResponse,
    passed: bool,
    weights: dict[str, float],
) -> float:
    """
    Compute the tested-implementation reward, given whether the code passed evaluation.

    Only the components present in `weights` contribute, each scaled by its weight. Each
    `if` below is independent, so a response can earn any combination of components. This
    is the execution-verified reward used for the stdin/stdout dataset (data/train/skywork).

    Returns a scalar reward.
    """
    reward = 0.0

    if weight := weights.get("correct_format"):
        # response has the expected shape: reasoning, an answer, and a code fragment
        # (structure only, not whether the code is valid python)
        if parsed.reasoning.strip() and parsed.answer and parsed.code:
            reward += weight

    if weight := weights.get("passes_syntax_check"):
        # the selected code fragment parses as valid python
        if parsed.valid:
            reward += weight

    if weight := weights.get("passed_tests"):
        # the code passed evaluation against the record's tests
        if passed:
            reward += weight

    # stdlib imports, excluding the io-boilerplate exception (see base.py)
    internal_libs = [lib for lib in parsed.std_libs if lib != _IO_BOILERPLATE_LIB]

    if weight := weights.get("internal_lib_imported"):
        # imports a stdlib module (other than the io boilerplate exception)
        if internal_libs:
            reward += weight

    if weight := weights.get("external_lib_imported"):
        # imports a third-party module
        if parsed.ext_libs:
            reward += weight

    if weight := weights.get("internal_lib_used"):
        # actually calls/uses an imported stdlib module
        if any(lib in parsed.lib_usage for lib in internal_libs):
            reward += weight

    if weight := weights.get("external_lib_used"):
        # actually calls/uses an imported third-party module
        if any(lib in parsed.lib_usage for lib in parsed.ext_libs):
            reward += weight

    return reward


class TestedImplementationReward(CodeReward):
    """
    Reward for code verified by executing it against stdin/stdout test cases.

    Runs the completion's code against the row's `inputs`/`expected_outputs` (the
    execution "pain point" this project otherwise avoids), then scores it with
    compute_reward. This is the reward used for the existing Skywork dataset.
    """

    # the class name starts with "Test", so tell pytest it is not a test class
    __test__ = False

    def score(
        self,
        content: str,
        row: dict[str, Any],
    ) -> float:
        """
        Score one completion by running its code against the row's tests.

        Returns a scalar reward.
        """
        processed = process_response(
            response=content,
            record={
                "inputs": row["inputs"],
                "expected_outputs": row["expected_outputs"],
            },
        )
        return compute_reward(
            parsed=processed,
            passed=processed.passed,
            weights=self.weights,
        )
