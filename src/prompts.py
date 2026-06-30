"""Build the system prompt for training and evaluation, from available library info."""

from pathlib import Path


# system prompt template used when a requirements file is provided - pinned version
# specs (e.g. "numpy==2.4.6") let the model infer which APIs are available
_SYSTEM_PROMPT_TEMPLATE = "You have access to a Python3.11 environment with the following libraries: {libraries}."


def load_requirements_libraries(
    requirements_file: str | Path,
) -> list[str]:
    """
    Parse the pinned library specs out of a requirements file (see
    setup/requirements/), returning them as "name==version" strings, e.g.
    "numpy==2.4.6" stays "numpy==2.4.6".

    Blank lines and comments are skipped.

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
) -> str:
    """
    Build the system prompt for runs that list available libraries.

    Lists the libraries in `requirements_file` (see setup/requirements/) so the model
    knows what it is able to import and reuse.

    Returns the system prompt string.
    """
    libraries = load_requirements_libraries(requirements_file)
    return _SYSTEM_PROMPT_TEMPLATE.format(libraries=", ".join(libraries))


def build_system_prompt(
    libraries_requirements_file: str | Path | None,
) -> str | None:
    """
    Build the effective system prompt for a training or evaluation run.

    When `libraries_requirements_file` is given, the prompt lists the available
    libraries (see build_libraries_system_prompt). When None, no system prompt is used.

    Returns the system prompt string, or None.
    """
    if libraries_requirements_file is None:
        return None

    return build_libraries_system_prompt(libraries_requirements_file)
