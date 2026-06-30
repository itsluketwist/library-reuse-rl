"""Tests for the available-libraries system-prompt builder."""

from pathlib import Path

from src.prompts import (
    build_libraries_system_prompt,
    build_system_prompt,
    load_requirements_libraries,
)


def _write_requirements(path: Path, lines: list[str]) -> Path:
    """Write `lines` to a requirements file at `path`."""
    requirements_file = path / "requirements.txt"
    requirements_file.write_text("\n".join(lines) + "\n")
    return requirements_file


class TestLoadRequirementsLibraries:
    """Tests for load_requirements_libraries."""

    def test_returns_full_pinned_specs(self, tmp_path: Path):
        """A pinned "name==version" line is returned as-is (version preserved)."""
        requirements_file = _write_requirements(
            tmp_path, ["numpy==2.4.6", "pandas==3.0.3"]
        )

        assert load_requirements_libraries(requirements_file) == [
            "numpy==2.4.6",
            "pandas==3.0.3",
        ]

    def test_skips_blank_lines_and_comments(self, tmp_path: Path):
        """Blank lines and comment lines don't produce library entries."""
        requirements_file = _write_requirements(
            tmp_path,
            ["# a comment", "", "numpy==2.4.6", "  ", "pandas==3.0.3"],
        )

        assert load_requirements_libraries(requirements_file) == [
            "numpy==2.4.6",
            "pandas==3.0.3",
        ]

    def test_preserves_file_order(self, tmp_path: Path):
        """Libraries are returned in the order they appear in the file."""
        requirements_file = _write_requirements(
            tmp_path, ["sympy==1.14.0", "numpy==2.4.6"]
        )

        assert load_requirements_libraries(requirements_file) == [
            "sympy==1.14.0",
            "numpy==2.4.6",
        ]


class TestBuildLibrariesSystemPrompt:
    """Tests for build_libraries_system_prompt."""

    def test_lists_every_library_with_version(self, tmp_path: Path):
        """The prompt includes each library's full pinned spec (name==version)."""
        requirements_file = _write_requirements(
            tmp_path, ["numpy==2.4.6", "pandas==3.0.3"]
        )

        prompt = build_libraries_system_prompt(requirements_file)

        assert "numpy==2.4.6" in prompt
        assert "pandas==3.0.3" in prompt


class TestBuildSystemPrompt:
    """Tests for build_system_prompt."""

    def test_returns_none_when_no_requirements_file(self, tmp_path: Path):
        """With libraries_requirements_file=None, no system prompt is built."""
        result = build_system_prompt(libraries_requirements_file=None)

        assert result is None

    def test_returns_full_prompt_when_requirements_file_given(self, tmp_path: Path):
        """With a requirements file, the prompt lists the available libraries."""
        requirements_file = _write_requirements(tmp_path, ["numpy==2.4.6"])

        result = build_system_prompt(libraries_requirements_file=requirements_file)

        assert result is not None
        assert "numpy==2.4.6" in result
