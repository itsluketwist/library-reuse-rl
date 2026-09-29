"""Tests for splitting reasoning from the answer and extracting python code blocks."""

import pytest
from lure import extract_code_blocks, split_reasoning


class TestSplitReasoning:
    """Tests for split_reasoning."""

    def test_open_and_close_tags(self):
        """Text between the tags is reasoning, and text after them is the answer."""
        assert split_reasoning("<think> plan </think>\nanswer") == ("plan", "answer")

    def test_close_tag_only(self):
        """Qwen3-style responses (opening tag in the prompt) still split correctly."""
        assert split_reasoning("plan\n</think>\n\nanswer") == ("plan", "answer")

    def test_open_tag_without_close_tag_has_no_answer(self):
        """A model still reasoning when the response ended has no answer."""
        assert split_reasoning("<think>still going") == ("still going", "")

    def test_no_tags_is_all_answer(self):
        """A response without reasoning tags is entirely answer."""
        assert split_reasoning("  just an answer  ") == ("", "just an answer")

    @pytest.mark.parametrize("tag", ["think", "thinking", "reasoning", "THOUGHT"])
    def test_tag_variants(self, tag: str):
        """The common reasoning tag names are recognised, in any case."""
        assert split_reasoning(f"<{tag}>a</{tag}>b") == ("a", "b")

    def test_only_first_close_tag_splits(self):
        """A later closing tag stays in the answer."""
        assert split_reasoning("a</think>b</think>c") == ("a", "b</think>c")


class TestCustomReasoningTags:
    """Tests for split_reasoning with custom reasoning tags."""

    def test_non_xml_tags(self):
        """Tags don't need to be xml-style, and are matched literally."""
        tags = [("[THINK]", "[/THINK]")]

        assert split_reasoning("[THINK]plan[/THINK]answer", tags) == ("plan", "answer")

    def test_single_pair(self):
        """A single (opening, closing) pair can be given without wrapping it in a list."""
        tags = ("<|begin_of_thought|>", "<|end_of_thought|>")

        assert split_reasoning("<|begin_of_thought|>a<|end_of_thought|>b", tags) == (
            "a",
            "b",
        )

    def test_regex_characters_are_literal(self):
        """Characters like | and [ in a tag never act as regex syntax."""
        assert split_reasoning("a|b", [("<|x|>", "|")]) == ("a", "b")

    def test_close_tag_only_and_unfinished(self):
        """Custom tags follow the same rules as the default ones."""
        tags = [("[THINK]", "[/THINK]")]

        assert split_reasoning("plan[/THINK]answer", tags) == ("plan", "answer")
        assert split_reasoning("[THINK]still going", tags) == ("still going", "")

    def test_custom_tags_replace_the_defaults(self):
        """With custom tags, the default <think> tags are no longer recognised."""
        tags = [("[THINK]", "[/THINK]")]

        assert split_reasoning("a</think>b", tags) == ("", "a</think>b")

    def test_no_tags(self):
        """With no tags, the whole response is the answer."""
        assert split_reasoning("a</think>b", []) == ("", "a</think>b")

    def test_matches_ignore_case(self):
        """Custom tags match in any case, like the defaults."""
        assert split_reasoning("a[/think]b", [("[THINK]", "[/THINK]")]) == ("a", "b")

    @pytest.mark.parametrize(
        "tags",
        [
            [("<think>",)],
            [("<think>", "")],
            ["</think>"],
            ("a", "b", "c"),
        ],
    )
    def test_bad_tags_raise(self, tags):
        """Tags that aren't (opening, closing) pairs of non-empty strings are rejected."""
        with pytest.raises(ValueError, match="pairs of strings"):
            split_reasoning("x", tags)


class TestExtractCodeBlocks:
    """Tests for extract_code_blocks."""

    def test_python_fences_in_order(self):
        """Every python fence is extracted, in response order."""
        answer = "```python\na = 1\n```\ntext\n```py\nb = 2\n```"
        assert extract_code_blocks(answer) == ["a = 1", "b = 2"]

    @pytest.mark.parametrize("info", ["", "python", "py", "python3", "Python title=x"])
    def test_python_languages(self, info: str):
        """Unlabelled and python-labelled fences (including extra info) are python."""
        assert extract_code_blocks(f"```{info}\nx = 1\n```") == ["x = 1"]

    def test_broken_python_labelled_block_is_kept(self):
        """A block labelled python is python, even when its code is broken."""
        assert extract_code_blocks("```python\ndef broken(:\n```") == ["def broken(:"]

    @pytest.mark.parametrize(
        "text",
        [
            "pip install requests\npython main.py",
            "project/\n├── main.py\n└── utils.py",
            "Output: 42 items processed",
        ],
    )
    def test_unlabelled_non_python_is_skipped(self, text: str):
        """An unlabelled fence only counts as python when its code parses."""
        assert extract_code_blocks(f"```\n{text}\n```") == []

    @pytest.mark.parametrize("info", ["bash", "sh", "javascript", "text", "json"])
    def test_other_languages_are_skipped(self, info: str):
        """Fences in other languages are never analysed as python."""
        assert extract_code_blocks(f"```{info}\nx = 1\n```") == []

    def test_empty_fence_is_skipped(self):
        """A fence with no code produces no block."""
        assert extract_code_blocks("```python\n\n```") == []

    def test_unterminated_fence_gives_nothing(self):
        """A truncated fence gets no credit, even when its code is valid."""
        assert extract_code_blocks("```python\nimport pandas\npandas.read_csv(p)") == []

    def test_complete_fence_before_truncated_fence(self):
        """Complete fences are kept and a trailing truncated fence is ignored."""
        answer = "```python\na = 1\n```\nthen\n```python\nb = "
        assert extract_code_blocks(answer) == ["a = 1"]

    def test_raw_python_fallback(self):
        """An unfenced answer that is entirely valid python is one raw block."""
        answer = "import numpy as np\nprint(np.mean(values))"
        assert extract_code_blocks(answer) == [answer]

    @pytest.mark.parametrize("answer", ["Use pandas for this.", "", "   "])
    def test_prose_is_not_code(self, answer: str):
        """Prose that isn't valid python never becomes a code block."""
        assert extract_code_blocks(answer) == []
