"""Build the system prompt for training and evaluation, from available library info."""

from pathlib import Path
from typing import Literal


# system prompt styles: "lib" lists the installed libraries, "use" also pushes the model
# to prefer them, "std" pushes the model to prefer the standard library, "ext" pushes it
# to prefer any third-party library, "base" skips the system prompt entirely (see
# build_system_prompt)
PromptStyle = Literal["lib", "use", "std", "ext", "base"]

# templates for the styles that list the requirements-file libraries (lib/use); the
# {libraries} placeholder is filled by build_libraries_system_prompt
_SYSTEM_PROMPT_TEMPLATES: dict[PromptStyle, str] = {
    "lib": (
        "You have access to a Python3.11 environment with the following "
        "libraries: {libraries}."
    ),
    "use": (
        "You have access to a Python3.11 environment with the following "
        "libraries: {libraries}. Actively prefer using these libraries over "
        "writing your own implementations - before writing any non-trivial logic, "
        "check whether one of these libraries already provides it."
    ),
}

# the styles whose prompt is fixed, independent of the requirements file, so they name no
# specific libraries (see build_system_prompt). "std" prefers the standard library; "ext"
# is its mirror, a generic nudge towards reusing a third-party library
FIXED_SYSTEM_PROMPTS: dict[PromptStyle, str] = {
    "std": (
        "You are writing Python3.11 code. Actively prefer using the Python standard "
        "library over writing your own implementations - before writing any "
        "non-trivial logic, check whether the standard library already provides it."
    ),
    "ext": (
        "You are writing Python3.11 code. Actively prefer using well-established "
        "third-party libraries over writing your own implementations - before writing "
        "any non-trivial logic, check whether a library already provides it."
    ),
}


def load_requirements_libraries(
    requirements_file: str | Path,
) -> list[str]:
    """
    Parse the pinned library specs out of a requirements file (see setup/requirements/),
    returning them as "name==version" strings. Blank lines and comments are skipped.

    Returns the pinned specs, in file order.
    """
    libraries = []
    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            libraries.append(line)

    return libraries


def build_libraries_system_prompt(
    requirements_file: str | Path,
    style: PromptStyle = "lib",
) -> str:
    """
    Build the system prompt for runs that list available libraries.

    Lists the libraries in `requirements_file` so the model knows what it can reuse.
    `style` selects the wording (see _SYSTEM_PROMPT_TEMPLATES): "lib" lists them, "use"
    also pushes the model to prefer them. "base" is filtered out by build_system_prompt
    before it reaches here.

    Returns the system prompt string.
    """
    libraries = load_requirements_libraries(requirements_file)
    return _SYSTEM_PROMPT_TEMPLATES[style].format(libraries=", ".join(libraries))


def build_system_prompt(
    libraries_requirements_file: str | Path | None,
    style: PromptStyle = "lib",
) -> str | None:
    """
    Build the effective system prompt for a training or evaluation run.

    "base" always returns None (no system prompt). "std"/"ext" return their fixed prompt
    (see FIXED_SYSTEM_PROMPTS) regardless of `libraries_requirements_file`, since they
    name no specific libraries. "lib"/"use" list the file's libraries (see
    build_libraries_system_prompt), and return None when no file is given.

    Returns the system prompt string, or None.
    """
    if style == "base":
        return None

    if style in FIXED_SYSTEM_PROMPTS:
        return FIXED_SYSTEM_PROMPTS[style]

    if libraries_requirements_file is None:
        return None

    return build_libraries_system_prompt(libraries_requirements_file, style=style)


def library_prompt_suffix(
    libraries_requirements_file: str | Path | None,
    style: PromptStyle = "lib",
) -> str:
    """
    Compute the output-path suffix for a run's system-prompt setting, e.g. "-libv1"/
    "-usev1" for requirements file "v1.txt", or a fixed "-std"/"-ext" for those styles.

    Used by eval_results_path so runs under different (file, style) combinations land in
    distinct output paths.

    Returns "" when `style` is "base", or when "lib"/"use" have no requirements file.
    """
    if style in FIXED_SYSTEM_PROMPTS:
        # file-independent, so a fixed suffix (no requirements-file version stem)
        return f"-{style}"

    if style == "base" or libraries_requirements_file is None:
        return ""

    return f"-{style}{Path(libraries_requirements_file).stem}"
