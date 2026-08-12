"""Tests for the shared library analysis in bcbreuse.libraries (and the build hard-exclude)."""

from bcbreuse.libraries import (
    analyse_source,
    extract_code,
    find_named_libraries,
    imported_libraries,
    pip_package,
    required_pip_packages,
)
from build import hard_exclude_reason


class TestPipPackage:
    """Import names map to their pip distribution names where they differ."""

    def test_mapped_name(self) -> None:
        assert pip_package("bs4") == "beautifulsoup4"

    def test_unmapped_name_passthrough(self) -> None:
        assert pip_package("numpy") == "numpy"


class TestAnalyseSource:
    """Source analysis splits standard-library from external imports."""

    def test_splits_std_and_external(self) -> None:
        code = "import json\nimport pandas as pd\n\ndef f(x):\n    return pd.Series(x).sum()\n"
        std, ext = analyse_source(code)
        assert "json" in std
        assert "pandas" in ext


class TestExtractAndImports:
    """Code extraction and import scanning of a model response."""

    _RESPONSE = "Sure!\n```python\nimport re\nprint(re.findall('a', input()))\n```\n"

    def test_extract_code_picks_fenced_block(self) -> None:
        code = extract_code(self._RESPONSE)
        assert code is not None and "re.findall" in code

    def test_extract_code_picks_largest_fenced_block(self) -> None:
        """Executable BCBReuse answers select one largest candidate program."""
        response = """```python
print("short")
```
```python
line = input()
value = line.strip()
print(value)
```"""

        code = extract_code(response)

        assert code is not None and "value = line.strip()" in code
        assert 'print("short")' not in code

    def test_extract_code_none_when_no_code(self) -> None:
        assert extract_code("just prose, no code here") is None

    def test_imported_libraries_excludes_stdlib(self) -> None:
        # `re` is stdlib, so it is not an external library
        assert imported_libraries(self._RESPONSE) == set()

    def test_required_pip_packages_maps_external(self) -> None:
        response = "```python\nimport bs4\nprint(bs4)\n```"
        assert required_pip_packages(response) == {"beautifulsoup4"}

    def test_required_pip_packages_honours_pip_install_line(self) -> None:
        response = "```bash\npip install rich\n```\n```python\nimport os\n```"
        assert "rich" in required_pip_packages(response)


class TestFindNamedLibraries:
    """The agnostic-prompt scan flags any prompt that still names a library."""

    def test_flags_named_library(self) -> None:
        assert "pandas" in find_named_libraries("Load the CSV into a pandas DataFrame.")

    def test_clean_prompt_has_none(self) -> None:
        assert (
            find_named_libraries("Read integers from stdin and print their sum.") == []
        )


class TestHardExcludeReason:
    """The build hard-exclude keeps most problems and drops only the impossible ones."""

    def test_network_dropped(self) -> None:
        assert hard_exclude_reason(["requests"]) == "network"

    def test_plot_dropped(self) -> None:
        assert hard_exclude_reason(["matplotlib", "numpy"]) == "plot-gui"

    def test_ordinary_libs_kept(self) -> None:
        assert hard_exclude_reason(["pandas", "numpy"]) is None
