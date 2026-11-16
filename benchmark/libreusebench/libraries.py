"""Extract and classify the libraries a model response (or a solution) uses.

Shared by the build pipeline (to analyse BigCodeBench reference solutions and scan rewritten
prompts) and by the evaluation harness (to find the code to run and the packages to install).
The parsing and library analysis is done by lure (Library-Use Rewards), the same package
the project's training rewards use.
"""

import re

from lure import CodeBlock, analyse_code, parse_response, pip_package
from lure.analysis import STDLIB_MODULES


# python standard-library module names, to tell stdlib imports from third-party ones
PYTHON_STDLIB: frozenset[str] = STDLIB_MODULES

# matches libraries named in a `pip install ...` line, so an explicit install the model wrote
# is honoured in addition to what its code imports
_PIP_INSTALL_REGEX = re.compile(
    r"[\n`]pip[^\S\n\r]+install[^\S\n\r]+([A-Za-z0-9_.\-]+(?:[^\S\n\r]+[A-Za-z0-9_.\-]+)*)",
)

# library names / spellings that must not appear in a rewritten prompt (it is meant to be
# library-agnostic); a hit means the prompt still leaks a specific library
_LIBRARY_NAME_TERMS = {
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "scikit-learn",
    "statsmodels",
    "dateutil",
    "pytz",
    "beautifulsoup",
    "bs4",
    "lxml",
    "requests",
    "flask",
    "django",
    "nltk",
    "regex",
    "pyyaml",
    "pillow",
    "opencv",
    "tensorflow",
    "pytorch",
    "torch",
    "keras",
}


def analyse_source(
    code: str,
) -> tuple[list[str], list[str]]:
    """Classify the standard-library and external imports of a python source string.

    Returns a (std_libs, ext_libs) tuple of sorted import names.
    """
    block = analyse_code(code)
    return sorted(block.stdlib_imports), sorted(block.external_imports)


def select_code_block(
    response: str,
) -> CodeBlock | None:
    """Select the code fragment to run/analyse from a model response.

    Any reasoning (e.g. <think>...</think>) is removed first, then the longest python code
    block in the answer is picked. If the answer has no code fences, the whole answer is used
    when it is valid python (see lure.parse_response).

    Returns the selected (already-analysed) CodeBlock, or None if the response has no code.
    """
    return parse_response(response).longest_block


def extract_code(
    response: str,
) -> str | None:
    """The candidate program to execute from a model response.

    Returns the selected code, or None if no code was found.
    """
    block = select_code_block(response)
    return block.code if block is not None else None


def imported_libraries(
    response: str,
) -> set[str]:
    """The external (non-stdlib) libraries the response's code imports.

    This counts any import, whether or not the code goes on to call or access it - see
    `used_libraries` for the stricter, actually-used signal.

    Returns the external library import names.
    """
    block = select_code_block(response)
    return set(block.external_imports) if block is not None else set()


def used_libraries(
    response: str,
) -> set[str]:
    """The external (non-stdlib) libraries the response's code both imports and actually uses.

    A library only counts once the code references a name it imported, not merely imports it -
    this is the stricter reuse signal, since an unused import can inflate `imported_libraries`
    without the model reaching for the library at all.

    Returns the external library import names that are actually used.
    """
    block = select_code_block(response)
    return set(block.external_used) if block is not None else set()


def _declared_installs(
    response: str,
) -> set[str]:
    """Libraries named in `pip install ...` lines in the response text.

    Returns the declared install names.
    """
    installs: set[str] = set()
    for match in _PIP_INSTALL_REGEX.findall(response):
        installs.update(match.strip().split())
    return installs


def required_import_names(
    response: str,
) -> set[str]:
    """The third-party import names a response needs, before mapping to pip distribution names.

    Combines the external libraries the code imports with any the response explicitly asks to
    `pip install`, and drops standard-library names.

    Returns the set of import names.
    """
    wanted = imported_libraries(response) | _declared_installs(response)
    return {name for name in wanted if name not in PYTHON_STDLIB}


def required_pip_packages(
    response: str,
) -> set[str]:
    """The pip packages the harness must install to run a response.

    Maps each required import name to its pip distribution name (e.g. sklearn -> scikit-learn).

    Returns the set of pip package names to install.
    """
    return {pip_package(name) for name in required_import_names(response)}


def find_named_libraries(
    text: str,
) -> list[str]:
    """Library names a text mentions (case-insensitive substring scan).

    Used to reject a rewritten prompt that still names a specific library, defeating the
    library-agnostic design.

    Returns the matched library terms, in a stable sorted order.
    """
    lowered = text.lower()
    return sorted(term for term in _LIBRARY_NAME_TERMS if term in lowered)
