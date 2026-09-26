"""Tests for mapping import names to PyPI package names."""

import pytest
from lure import pip_package


@pytest.mark.parametrize(
    ("import_name", "package"),
    [
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
        ("yaml", "PyYAML"),
        ("pandas", "pandas"),
    ],
)
def test_pip_package(import_name: str, package: str):
    """Import names map to their package name, or stay the same when they match."""
    assert pip_package(import_name) == package
