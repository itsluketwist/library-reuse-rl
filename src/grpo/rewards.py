"""Reward computation for GRPO training, built on the response parsing/evaluation pipeline."""

import functools
import operator
from typing import Callable

from src.response.parser import ParsedResponse
from src.response.pipeline import process_response
from src.utils.enums import WeightedIntFlag


# "os" is commonly imported only for fast stdin/stdout reading (e.g. os.read(0, ...)),
# not as a genuine example of reusing library functionality, so it's excluded from
# the internal-library bits below.
_IO_BOILERPLATE_LIB = "os"


class RewardBit(WeightedIntFlag):
    """
    Bitmask flags for the reward components a response can satisfy, each carrying
    the weight it contributes when satisfied and switched on.

    A caller (e.g. the training cli) passes a combination of these bits in as the
    `reward_bits` argument to compute_reward, to choose which components actually
    count towards the reward - this is what lets the reward function itself be a
    configurable choice at training time, rather than fixed in code.

    New components should be added as new members, written below as just their
    weight (a placeholder, since the RL approach isn't decided yet) - see
    WeightedIntFlag for how the actual bit value (1, 2, 4, ...) gets auto-assigned.
    """

    CORRECT_FORMAT = 0.1  # answer contains a non-empty, valid python fragment
    PASSED_TESTS = 1.0  # code passed evaluation against the record's tests
    INTERNAL_LIB_IMPORTED = 0.05  # imports a stdlib module (other than os)
    EXTERNAL_LIB_IMPORTED = 0.05  # imports a third-party module
    INTERNAL_LIB_USED = 0.05  # actually calls/uses an imported stdlib module
    EXTERNAL_LIB_USED = 0.05  # actually calls/uses an imported third-party module


# every bit or-ed together - the default `reward_bits`, so all components count
# unless a caller deliberately restricts which ones are switched on
ALL_REWARD_BITS: RewardBit = functools.reduce(operator.or_, RewardBit, RewardBit(0))


def parse_reward_bits(value: str) -> RewardBit:
    """
    Parse a comma-separated list of RewardBit names (e.g. "CORRECT_FORMAT,PASSED_TESTS")
    into a bitmask - this is what lets a config file or command line flag switch
    individual reward components on/off by name, rather than needing to work out a raw
    integer bitmask by hand. A blank string parses to no bits switched on.

    Returns the combined RewardBit bitmask.
    """
    value = value.strip()
    if not value:
        return RewardBit(0)

    bits = RewardBit(0)
    for name in value.split(","):
        bits |= RewardBit[name.strip()]
    return bits


def compute_reward_bits(
    parsed: ParsedResponse,
    passed: bool,
) -> RewardBit:
    """
    Determine which reward components a response satisfies.

    This is unconditional on `reward_bits` - it reports everything the response
    satisfies, regardless of which components a given training run has switched on.

    Returns the bitmask of satisfied RewardBit flags.
    """
    bits = RewardBit(0)

    if parsed.valid and parsed.code.strip():
        bits |= RewardBit.CORRECT_FORMAT

    if passed:
        bits |= RewardBit.PASSED_TESTS

    internal_libs = [lib for lib in parsed.std_libs if lib != _IO_BOILERPLATE_LIB]
    if internal_libs:
        bits |= RewardBit.INTERNAL_LIB_IMPORTED

    if parsed.ext_libs:
        bits |= RewardBit.EXTERNAL_LIB_IMPORTED

    if any(lib in parsed.lib_usage for lib in internal_libs):
        bits |= RewardBit.INTERNAL_LIB_USED

    if any(lib in parsed.lib_usage for lib in parsed.ext_libs):
        bits |= RewardBit.EXTERNAL_LIB_USED

    return bits


def compute_reward(
    parsed: ParsedResponse,
    passed: bool,
    reward_bits: RewardBit = ALL_REWARD_BITS,
) -> float:
    """
    Compute a reward for a response, given whether its code passed evaluation.

    Only the components switched on in `reward_bits` contribute - this is how a
    training run chooses its reward function (e.g. correctness-only, versus
    correctness plus library-reuse shaping) without changing this code.

    Returns a scalar reward.
    """
    satisfied = compute_reward_bits(parsed=parsed, passed=passed) & reward_bits
    return sum(bit.weight for bit in RewardBit if bit in satisfied)


def make_library_reuse_reward(
    reward_bits: RewardBit,
) -> Callable[..., list[float]]:
    """
    Build a reward function bound to `reward_bits`, for GRPOTrainer's `reward_funcs`.

    Returns a callable scoring each completion with the project's library-reuse
    reward, restricted to the components switched on in `reward_bits`.
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
                    reward_bits=reward_bits,
                )
            )

        return rewards

    return library_reuse_reward
