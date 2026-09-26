"""Tests for the TRL GRPOTrainer adapter."""

import pytest
from lure import create_library_reward, create_trl_reward, stop_reason_from_tokens


_EOS = 99
_RESPONSE = "<think>r</think>\n```python\nimport requests\nrequests.get(u)\n```"


class TestStopReasonFromTokens:
    """Tests for stop_reason_from_tokens."""

    @pytest.mark.parametrize(
        ("ids", "eos", "expected"),
        [
            ([1, 2, _EOS], _EOS, "stop"),
            ([1, 2, 3], _EOS, "length"),
            ([1, 2, 7], [_EOS, 7], "stop"),
            (None, _EOS, None),
            ([], _EOS, None),
            ([1, _EOS], None, None),
        ],
    )
    def test_stop_reason(self, ids, eos, expected):
        """Ending on an eos token is a natural stop, otherwise the limit was hit."""
        assert stop_reason_from_tokens(ids, eos) == expected


class TestCreateTrlReward:
    """Tests for create_trl_reward."""

    def test_standard_and_conversational_completions(self):
        """Plain-string and chat-message completions score the same."""
        reward = create_trl_reward()

        rewards = reward(
            completions=[_RESPONSE, [{"role": "assistant", "content": _RESPONSE}]],
        )

        assert rewards == [2.0, 2.0]

    def test_truncation_from_completion_ids(self):
        """Completions not ending in eos lose the format component."""
        reward = create_trl_reward(eos_token_id=_EOS)

        rewards = reward(
            completions=[_RESPONSE, _RESPONSE],
            completion_ids=[[1, _EOS], [1, 2]],
        )

        assert rewards == [2.0, 1.75]

    def test_per_row_dataset_columns(self):
        """Each completion uses its own row's columns, and other columns are ignored."""
        reward = create_trl_reward(
            reward=create_library_reward(appropriate_lib_used=1.0),
            appropriate_libraries_column="useful_libs",
        )

        rewards = reward(
            completions=[_RESPONSE, _RESPONSE, _RESPONSE],
            local_modules=[[], ["requests"], None],
            useful_libs=[["requests"], ["requests"], ["numpy"]],
            trainer_state=object(),
        )

        assert rewards == [3.0, 0.5, 2.0]

    def test_per_item_names_from_id_column(self):
        """A reward created with per-item dicts reads each row's id from the dataset."""
        reward = create_trl_reward(
            reward=create_library_reward(local_modules={"a": ["requests"], "b": []}),
            item_id_column="task_id",
        )

        rewards = reward(
            completions=[_RESPONSE, _RESPONSE],
            task_id=["a", "b"],
        )

        assert rewards == [0.5, 2.0]

    def test_name(self):
        """TRL logs the reward under its name."""
        # getattr, since type checkers don't know a generic callable has a __name__
        assert getattr(create_trl_reward(), "__name__") == "library_reward"
