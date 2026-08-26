"""Extract and classify the libraries a model response (or a solution) uses.

Shared by the build pipeline (to analyse BigCodeBench reference solutions and scan rewritten
prompts) and by the evaluation harness (to find the code to run and the packages to install).
Only depends on llm_cgr's code analyser, so the released package stays light.
"""

import re
import sys
import warnings

from llm_cgr import CodeBlock, Markdown


# python standard-library module names, to tell stdlib imports from third-party ones
PYTHON_STDLIB: frozenset[str] = getattr(sys, "stdlib_module_names", frozenset())

# top-level import name -> pip distribution name, where they differ (detection keys off the
# import name, but the harness must `pip install` the distribution name)
IMPORT_TO_PIP: dict[str, str] = {
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "fitz": "PyMuPDF",
    "dotenv": "python-dotenv",
    "Crypto": "pycryptodome",
    "OpenSSL": "pyOpenSSL",
}

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


def pip_package(
    import_name: str,
) -> str:
    """The pip distribution name for a top-level import name.

    Returns the mapped pip name, or the import name unchanged when no mapping is needed.
    """
    return IMPORT_TO_PIP.get(import_name, import_name)


def analyse_source(
    code: str,
) -> tuple[list[str], list[str]]:
    """Classify the standard-library and external imports of a python source string.

    Returns a (std_libs, ext_libs) tuple.
    """
    # some code embeds regex in plain (non-raw) strings, so compiling it emits SyntaxWarnings we
    # don't care about here - silence them to keep output clean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        block = CodeBlock(language="python", text=code)
    return block.std_libs, block.ext_libs


def select_code_block(
    response: str,
) -> CodeBlock:
    """Select the code fragment to run/analyse from a model response.

    Picks the longest fenced code block; if the response has no fenced blocks, falls back to the
    whole response when it parses as valid python.

    Returns the selected (already-analysed) CodeBlock.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        blocks = Markdown(
            text=response, default_codeblock_language="python"
        ).code_blocks
        if blocks:
            return max(blocks, key=lambda block: len(block.text.splitlines()))
        whole = CodeBlock(language="python", text=response)
        return whole if whole.valid else CodeBlock(language="python", text="")


def extract_code(
    response: str,
) -> str | None:
    """The candidate program to execute from a model response.

    Returns the selected code, or None if no code was found.
    """
    return select_code_block(response).text or None


def imported_libraries(
    response: str,
) -> set[str]:
    """The external (non-stdlib) libraries the response's code imports.

    This counts any import, whether or not the code goes on to call or access it - see
    `used_libraries` for the stricter, actually-used signal.

    Returns the external library import names.
    """
    return set(select_code_block(response).ext_libs)


def used_libraries(
    response: str,
) -> set[str]:
    """The external (non-stdlib) libraries the response's code both imports and actually uses.

    A library only counts once the code calls or accesses one of its members, not merely
    imports it (`CodeBlock.lib_usage`) - this is the stricter reuse signal, since an unused
    import can inflate `imported_libraries` without the model reaching for the library at all.

    Returns the external library import names that are actually used.
    """
    block = select_code_block(response)
    return {name for name in block.ext_libs if name in block.lib_usage}


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
