"""Use lure rewards with TRL's GRPOTrainer (TRL itself is not a dependency)."""

from collections.abc import Callable, Iterable
from typing import Any

from lure.reward import LibraryReward, create_library_reward


def stop_reason_from_tokens(
    completion_ids: list[int] | None,
    eos_token_id: int | Iterable[int] | None,
) -> str | None:
    """
    Work out how a generation ended from its tokens, for trainers that give reward
    functions no finish reason (TRL only passes the completion's token ids).

    A completion that ends on an end-of-sequence token stopped naturally ("stop"), and
    one that doesn't hit the token limit ("length"). Several eos tokens can be given,
    since some models have more than one.

    Returns "stop" or "length", or None when either input is missing.
    """
    if eos_token_id is None or not completion_ids:
        return None
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)
    return "stop" if completion_ids[-1] in eos_ids else "length"


def _completion_text(
    completion: str | list[dict[str, str]],
) -> str:
    """
    Get the text of a TRL completion, which is either a plain string (standard format)
    or a list of chat messages (conversational format).

    Returns the completion text.
    """
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"] if completion else ""


def create_trl_reward(
    reward: LibraryReward | None = None,
    eos_token_id: int | Iterable[int] | None = None,
    local_modules_column: str = "local_modules",
    target_libraries_column: str = "target_libraries",
    item_id_column: str = "id",
) -> Callable[..., list[float]]:
    """
    Wrap a LibraryReward as a TRL GRPOTrainer reward function.

    TRL calls reward functions with a batch of completions, their token ids, and every
    dataset column as a keyword argument. Pass the tokenizer's `eos_token_id` so truncated
    completions can be detected. If the dataset has `local_modules` or
    `target_libraries` columns (the names can be changed), each row's values are used
    for its completion. If instead the reward was created with dicts keyed by item id, each
    row's id is read from the `item_id_column`. By default the reward uses
    create_library_reward's weights.

    Returns a function to pass in GRPOTrainer's `reward_funcs`.
    """
    reward = reward or create_library_reward()
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
    eos_ids = list(eos_ids) if eos_ids is not None else None

    def reward_function(
        completions: list[Any],
        completion_ids: list[list[int]] | None = None,
        **columns: Any,
    ) -> list[float]:
        """
        Score a batch of TRL completions.

        Returns one reward per completion.
        """
        local_modules = columns.get(local_modules_column)
        target_libraries = columns.get(target_libraries_column)
        item_ids = columns.get(item_id_column)

        rewards = []
        for index, completion in enumerate(completions):
            stop_reason = stop_reason_from_tokens(
                completion_ids[index] if completion_ids else None,
                eos_ids,
            )
            rewards.append(
                reward(
                    _completion_text(completion),
                    stop_reason=stop_reason,
                    item_id=item_ids[index] if item_ids else None,
                    local_modules=local_modules[index] if local_modules else None,
                    target_libraries=(
                        target_libraries[index] if target_libraries else None
                    ),
                ),
            )
        return rewards

    # trl logs each reward function under its __name__
    reward_function.__name__ = reward.__name__
    return reward_function
