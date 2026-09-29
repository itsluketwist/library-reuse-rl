"""Tests for parsing a whole response."""

from lure import parse_response


class TestParseResponse:
    """Tests for parse_response and ParsedResponse."""

    def test_reasoning_is_never_analysed(self):
        """Library code in the reasoning earns nothing - only the answer counts."""
        parsed = parse_response(
            "<think>import pandas\npandas.read_csv(path)</think>"
            "```python\nprint('final answer')\n```",
        )

        assert parsed.external_imports == set()
        assert parsed.valid

    def test_libraries_combine_across_blocks(self):
        """Libraries from every block are reported together."""
        parsed = parse_response(
            "```python\nimport requests\nrequests.get(u)\n```\n"
            "```python\nimport json\nimport yaml\n```",
        )

        assert parsed.external_imports == {"requests", "yaml"}
        assert parsed.external_used == {"requests"}
        assert parsed.stdlib_imports == {"json"}

    def test_no_usage_across_blocks(self):
        """An import in one block and a call in another is not usage."""
        parsed = parse_response(
            "```python\nimport pandas\n```\n```python\npandas.x()\n```"
        )

        assert parsed.external_imports == {"pandas"}
        assert parsed.external_used == set()

    def test_valid_needs_every_block_to_parse(self):
        """One broken block makes the whole answer invalid."""
        parsed = parse_response("```python\na = 1\n```\n```python\ndef broken(:\n```")

        assert not parsed.valid

    def test_unlabelled_output_block_does_not_affect_validity(self):
        """An unlabelled block of program output isn't python, so can't be invalid."""
        parsed = parse_response(
            "```python\nprint(sum([1, 2]))\n```\nThis prints:\n```\n3 items processed\n```",
        )

        assert parsed.valid
        assert len(parsed.code_blocks) == 1

    def test_long_bash_block_does_not_affect_validity(self):
        """A longer non-python block is ignored, rather than failing the syntax check."""
        parsed = parse_response(
            "```bash\npip install pandas\ncd x\nls\nls\nls\n```\n"
            "```python\nimport pandas as pd\npd.read_csv(p)\n```",
        )

        assert parsed.valid
        assert len(parsed.code_blocks) == 1

    def test_no_code_is_not_valid(self):
        """An answer without code has no blocks and isn't valid."""
        parsed = parse_response("I would use pandas for this task.")

        assert parsed.code_blocks == ()
        assert not parsed.valid
        assert parsed.longest_block is None

    def test_longest_block(self):
        """The longest block is chosen by line count."""
        parsed = parse_response("```python\na = 1\n```\n```python\nb = 1\nc = 2\n```")

        assert parsed.longest_block is not None
        assert parsed.longest_block.code == "b = 1\nc = 2"

    def test_none_response(self):
        """A missing response parses as empty, rather than raising."""
        parsed = parse_response(None)

        assert parsed.answer == ""
        assert parsed.code_blocks == ()

    def test_truncated(self):
        """Only the "length" stop reason marks a response as truncated."""
        assert parse_response("x", stop_reason="length").truncated
        assert not parse_response("x", stop_reason="stop").truncated
        assert not parse_response("x").truncated

    def test_custom_reasoning_tags(self):
        """Custom reasoning tags keep library code in the reasoning out of the analysis."""
        parsed = parse_response(
            "[THINK]import pandas\npandas.read_csv(p)[/THINK]```python\nx = 1\n```",
            reasoning_tags=[("[THINK]", "[/THINK]")],
        )

        assert parsed.reasoning == "import pandas\npandas.read_csv(p)"
        assert parsed.external_imports == set()

    def test_local_modules(self):
        """Local modules are passed through to every block's analysis."""
        parsed = parse_response(
            "```python\nimport helpers\nhelpers.solve()\n```",
            local_modules=["helpers"],
        )

        assert parsed.external_imports == set()
