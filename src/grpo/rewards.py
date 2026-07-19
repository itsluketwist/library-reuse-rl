"""Reward computation for GRPO training, built on the response parsing/evaluation pipeline."""

from typing import Callable

from src.response.parser import ParsedResponse
from src.response.pipeline import process_response
from src.utils.config import load_yaml_config


# excluded from "library reuse" - "os" is usually just stdin/stdout boilerplate
_IO_BOILERPLATE_LIB = "os"


# valid component names for a config/rewards.yaml profile - add new ones here,
# with a matching `if` branch in compute_reward below
REWARD_COMPONENTS = frozenset(
    {
        "correct_format",
        "passes_syntax_check",
        "passed_tests",
        "internal_lib_imported",
        "external_lib_imported",
        "internal_lib_used",
        "external_lib_used",
    }
)


def compute_reward(
    parsed: ParsedResponse,
    passed: bool,
    weights: dict[str, float],
) -> float:
    """
    Compute a reward for a response, given whether its code passed evaluation.

    Only the components present in `weights` contribute, each scaled by its weight -
    this is how a reward profile (config/rewards.yaml) chooses the reward function for
    a training run without changing this code. Each `if` below is independent, so a
    response can earn any combination of components.

    Returns a scalar reward.
    """
    reward = 0.0

    if weight := weights.get("correct_format"):
        # response followed the expected think/answer shape - non-empty reasoning, a
        # non-empty answer, and a code fragment found in it. this is about response
        # structure only, not whether the code itself is valid python. answer and
        # code are None when nothing was extracted, so truthiness covers both cases.
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

    # stdlib imports, excluding the io-boilerplate exception (see above)
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


def load_reward_profile(
    profile: str,
    config_file: str = "config/rewards.yaml",
) -> dict[str, float]:
    """
    Load a named reward profile from `config_file` (see config/rewards.yaml).

    Every key in the profile is validated against REWARD_COMPONENTS, so a typo'd
    component name fails fast at startup instead of silently never contributing.

    Returns the profile's {component_name: weight} mapping.
    """
    weights = load_yaml_config(config_file=config_file, key=profile)

    unknown = sorted(set(weights) - REWARD_COMPONENTS)
    if unknown:
        raise ValueError(
            f"Unknown reward component(s) {unknown} in profile '{profile}' of "
            f"{config_file}. Available components: {sorted(REWARD_COMPONENTS)}"
        )

    return weights


def make_library_reuse_reward(
    weights: dict[str, float],
) -> Callable[..., list[float]]:
    """
    Build a reward function bound to `weights`, for GRPOTrainer's `reward_funcs`.

    Returns a callable scoring each completion with the project's library-reuse
    reward, restricted to the components present in `weights`.
    """

    def library_reuse_reward(
        completions: list[list[dict[str, str]]],
        inputs: list[list[str]],
        expected_outputs: list[list[str]],
        **kwargs,
    ) -> list[float]:
        """
        Score each completion with the project's library-reuse reward.

        `inputs`/`expected_outputs` are the dataset columns process_response needs to
        evaluate a completion's code against its test cases - GRPOTrainer supplies
        every dataset column as its own keyword argument, one list per column, so any
        other (unused) columns just collect harmlessly into **kwargs.

        Returns one reward per completion.
        """
        contents = [completion[0]["content"] for completion in completions]

        rewards = []
        for content, _inputs, _outputs in zip(contents, inputs, expected_outputs):
            processed = process_response(
                response=content,
                record={
                    "inputs": _inputs,
                    "expected_outputs": _outputs,
                },
            )
            rewards.append(
                compute_reward(
                    parsed=processed,
                    passed=processed.passed,
                    weights=weights,
                )
            )

        return rewards

    return library_reuse_reward
