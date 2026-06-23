"""Tests for the GRPO reward function factory."""

from src.grpo.rewards import make_library_reuse_reward
from src.response.parser import ParsedResponse
from src.response.pipeline import ProcessedResponse
from src.response.reward import ALL_REWARD_BITS, RewardBit


def _fake_processed(reward: float) -> ProcessedResponse:
    """Build a ProcessedResponse for reward tests, only the reward value matters."""
    parsed = ParsedResponse(
        reasoning="",
        answer="",
        code="",
        valid=True,
        error=None,
        std_libs=[],
        ext_libs=[],
        lib_usage={},
        logical_lines=0,
        maintainability_index=0.0,
    )
    return ProcessedResponse(parsed=parsed, passed=True, output="", reward=reward)


class TestMakeLibraryReuseReward:
    """Tests for make_library_reuse_reward."""

    def test_rebuilds_record_from_kwargs_and_scores_each_completion(self, monkeypatch):
        """Each completion is paired with its own record, zipped back from the kwargs columns."""
        seen_calls = []

        def fake_process_response(response, record, reward_bits):
            seen_calls.append((response, record, reward_bits))
            return _fake_processed(reward=float(len(response)))

        monkeypatch.setattr("src.grpo.rewards.process_response", fake_process_response)

        completions = [
            [{"content": "aa"}],
            [{"content": "bbb"}],
        ]

        library_reuse_reward = make_library_reuse_reward(ALL_REWARD_BITS)
        rewards = library_reuse_reward(
            completions,
            inputs=[["1\n"], ["2\n"]],
            expected_outputs=[["1\n"], ["2\n"]],
        )

        assert rewards == [2.0, 3.0]
        assert seen_calls == [
            ("aa", {"inputs": ["1\n"], "expected_outputs": ["1\n"]}, ALL_REWARD_BITS),
            ("bbb", {"inputs": ["2\n"], "expected_outputs": ["2\n"]}, ALL_REWARD_BITS),
        ]

    def test_binds_the_given_reward_bits(self, monkeypatch):
        """The factory's reward_bits argument is passed through to process_response."""
        seen_reward_bits = []

        def fake_process_response(response, record, reward_bits):
            seen_reward_bits.append(reward_bits)
            return _fake_processed(reward=0.0)

        monkeypatch.setattr("src.grpo.rewards.process_response", fake_process_response)

        reward_bits = RewardBit.PASSED_TESTS | RewardBit.CORRECT_FORMAT
        library_reuse_reward = make_library_reuse_reward(reward_bits)
        library_reuse_reward(
            [[{"content": "aa"}]],
            inputs=[["1\n"]],
            expected_outputs=[["1\n"]],
        )

        assert seen_reward_bits == [reward_bits]
