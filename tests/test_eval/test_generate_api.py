"""Tests for API-served generation (src/eval/generate_api.py), mocking the OpenAI client
so these run without network access or real credentials."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.eval.generate_api import ApiSettings, generate_api_responses


DEEPSEEK = ApiSettings(
    api_key_env="FAKE_API_KEY",
    base_url="https://api.deepseek.com",
)
OPENAI = ApiSettings(api_key_env="FAKE_API_KEY")


def _fake_completion(
    content: str,
    finish_reason: str,
    reasoning: str | None = None,
) -> SimpleNamespace:
    """Build a minimal stand-in for an OpenAI ChatCompletion response."""
    message = SimpleNamespace(content=content)
    if reasoning is not None:
        message.reasoning_content = reasoning
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class TestGenerateApiResponses:
    """Tests for generate_api_responses."""

    def test_maps_responses_and_stop_reasons_in_order(self, monkeypatch):
        """Each prompt's response and finish_reason come back in the same order,
        with finish_reason mapped through this project's stop_reason vocabulary."""
        monkeypatch.setenv("FAKE_API_KEY", "test-key")

        completions_by_prompt = {
            "hello": _fake_completion("hi there", "stop"),
            "count to a lot": _fake_completion("1 2 3...", "length"),
        }

        with patch("src.eval.generate_api.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.side_effect = (
                lambda model, messages, **kwargs: completions_by_prompt[
                    messages[0]["content"]
                ]
            )

            responses = generate_api_responses(
                model="deepseek-v4-flash",
                api=DEEPSEEK,
                prompts=[
                    [{"role": "user", "content": "hello"}],
                    [{"role": "user", "content": "count to a lot"}],
                ],
                decoding_settings={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "max_new_tokens": 128,
                },
            )

        assert responses == [[("hi there", "eos")], [("1 2 3...", "length")]]
        mock_openai_cls.assert_called_once_with(
            api_key="test-key", base_url="https://api.deepseek.com", max_retries=5
        )

    def test_n_samples_requests_one_call_per_sample_per_prompt(self, monkeypatch):
        """decoding_settings' n_samples requests that many independent completions per
        prompt (not the API's own `n` param, which not every provider supports), and
        groups them back under their prompt in order."""
        monkeypatch.setenv("FAKE_API_KEY", "test-key")

        completions = [
            _fake_completion("a1", "stop"),
            _fake_completion("a2", "stop"),
            _fake_completion("a3", "length"),
            _fake_completion("b1", "stop"),
            _fake_completion("b2", "stop"),
            _fake_completion("b3", "stop"),
        ]

        with patch("src.eval.generate_api.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.side_effect = completions

            responses = generate_api_responses(
                model="deepseek-v4-flash",
                api=DEEPSEEK,
                prompts=[
                    [{"role": "user", "content": "prompt a"}],
                    [{"role": "user", "content": "prompt b"}],
                ],
                decoding_settings={
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_new_tokens": 128,
                    "n_samples": 3,
                },
                max_workers=1,  # keep requests in submission order for this assertion
            )

        assert mock_client.chat.completions.create.call_count == 6
        assert responses == [
            [("a1", "eos"), ("a2", "eos"), ("a3", "length")],
            [("b1", "eos"), ("b2", "eos"), ("b3", "eos")],
        ]

    def test_missing_api_key_env_raises(self, monkeypatch):
        """A clear error is raised when the configured api_key_env is unset, rather than
        an opaque auth failure from the SDK."""
        monkeypatch.delenv("MISSING_KEY", raising=False)

        with pytest.raises(RuntimeError, match="MISSING_KEY"):
            generate_api_responses(
                model="gpt-5.6-luna",
                api=ApiSettings(api_key_env="MISSING_KEY"),
                prompts=[[{"role": "user", "content": "hi"}]],
                decoding_settings={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 128,
                },
            )

    def test_top_k_is_not_forwarded_to_the_api(self, monkeypatch):
        """decoding_settings' top_k (a vllm-only knob) is silently dropped rather than
        passed through to an API that doesn't accept it."""
        monkeypatch.setenv("FAKE_API_KEY", "test-key")

        with patch("src.eval.generate_api.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.return_value = _fake_completion(
                "ok", "stop"
            )

            generate_api_responses(
                model="gpt-5.6-luna",
                api=OPENAI,
                prompts=[[{"role": "user", "content": "hi"}]],
                decoding_settings={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 20,
                    "max_new_tokens": 64,
                },
            )

            _, kwargs = mock_client.chat.completions.create.call_args
            assert "top_k" not in kwargs
            assert kwargs["temperature"] == 0.7
            assert kwargs["top_p"] == 0.9
            assert kwargs["max_tokens"] == 64

    def test_extra_params_forwarded_via_extra_body(self, monkeypatch):
        """extra_params (e.g. DeepSeek's "thinking" toggle - see config/models.yaml) is
        sent through the SDK's extra_body, since it isn't part of the standard schema."""
        monkeypatch.setenv("FAKE_API_KEY", "test-key")

        with patch("src.eval.generate_api.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.return_value = _fake_completion(
                "ok", "stop"
            )

            generate_api_responses(
                model="deepseek-v4-flash",
                api=DEEPSEEK._replace(extra_params={"thinking": {"type": "disabled"}}),
                prompts=[[{"role": "user", "content": "hi"}]],
                decoding_settings={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 64,
                },
            )

            _, kwargs = mock_client.chat.completions.create.call_args
            assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_max_tokens_param_overrides_the_request_field_name(self, monkeypatch):
        """max_tokens_param (e.g. "max_completion_tokens" for newer OpenAI reasoning
        models - see config/models.yaml) selects which field the token cap is sent
        under, instead of always sending the legacy "max_tokens"."""
        monkeypatch.setenv("FAKE_API_KEY", "test-key")

        with patch("src.eval.generate_api.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.return_value = _fake_completion(
                "ok", "stop"
            )

            generate_api_responses(
                model="gpt-5.6-luna",
                api=OPENAI._replace(max_tokens_param="max_completion_tokens"),
                prompts=[[{"role": "user", "content": "hi"}]],
                decoding_settings={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 64,
                },
            )

            _, kwargs = mock_client.chat.completions.create.call_args
            assert kwargs["max_completion_tokens"] == 64
            assert "max_tokens" not in kwargs

    def test_reasoning_content_folded_into_think_tags(self, monkeypatch):
        """Reasoning models return their thinking in a separate field; it is folded back
        into the response text as a <think> block so strip_reasoning and the n_reasoning
        metric see the same shape they do from vllm."""
        monkeypatch.setenv("FAKE_API_KEY", "test-key")

        with patch("src.eval.generate_api.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.return_value = _fake_completion(
                "the answer", "stop", reasoning="let me think"
            )

            responses = generate_api_responses(
                model="deepseek-v4-flash",
                api=DEEPSEEK,
                prompts=[[{"role": "user", "content": "hi"}]],
                decoding_settings={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 64,
                },
            )

        assert responses == [[("<think>let me think</think>the answer", "eos")]]

    def test_failed_request_does_not_lose_the_other_samples(self, monkeypatch):
        """A request that fails even after the SDK's own retries is recorded against that
        one sample, rather than raising and discarding every other completed generation."""
        monkeypatch.setenv("FAKE_API_KEY", "test-key")

        completions = [
            _fake_completion("fine", "stop"),
            RuntimeError("502 bad gateway"),
        ]

        with patch("src.eval.generate_api.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value
            mock_client.chat.completions.create.side_effect = completions

            responses = generate_api_responses(
                model="deepseek-v4-flash",
                api=DEEPSEEK,
                prompts=[
                    [{"role": "user", "content": "a"}],
                    [{"role": "user", "content": "b"}],
                ],
                decoding_settings={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 64,
                },
                max_workers=1,
            )

        assert responses[0] == [("fine", "eos")]
        failed_text, failed_stop_reason = responses[1][0]
        assert failed_stop_reason == "error"
        assert "502 bad gateway" in failed_text
