"""Build the GRPO reward function from the project's library-reuse reward pipeline."""

from typing import Callable

from src.response.pipeline import process_response
from src.response.reward import RewardBit


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

        return [
            process_response(
                response=content,
                record={
                    "inputs": _inputs,
                    "expected_outputs": _outputs,
                },
                reward_bits=reward_bits,
            ).reward
            for content, _inputs, _outputs in zip(contents, inputs, expected_outputs)
        ]

    return library_reuse_reward
