#!/usr/bin/env bash

# creates the isolated venv used to run llm-generated code during evaluation, kept
# separate from the project's own venv so we control exactly which libraries
# generated code is able to import. takes the requirements version to install,
# matching one of the files in setup/requirements/ (e.g. "v1" for v1.txt).
#
# after installing, the requirements file is rewritten with the exact versions
# that were installed, so it doubles as a pinned record of what was used.
#
# usage: bash setup/setup_eval_venv.sh v1

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ $# -ne 1 ]; then
    echo "usage: $0 <requirements-version>" >&2
    echo "available versions:" >&2
    ls setup/requirements >&2
    exit 1
fi

REQUIREMENTS_FILE="setup/requirements/$1.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "no requirements file found at ${REQUIREMENTS_FILE}" >&2
    exit 1
fi

# remove any existing eval venv first, so rerunning always gives a clean rebuild
# rather than layering new packages on top of whatever was already installed
rm -rf .venv-eval

uv venv .venv-eval --python 3.12
uv pip install --python .venv-eval -r "$REQUIREMENTS_FILE"

# pin the requirements file to the exact versions that were just installed, so it
# becomes a record of what was actually used rather than just a list of names
python3 - "$REQUIREMENTS_FILE" <(uv pip freeze --python .venv-eval) <<'PYEOF'
import re
import sys

requirements_path, freeze_path = sys.argv[1], sys.argv[2]


def normalise(name: str) -> str:
    """Normalise a package name (PEP 503) so freeze output can be matched back to it."""
    return re.sub(r"[-_.]+", "-", name).lower()


with open(freeze_path) as f:
    versions = {
        normalise(name): version
        for name, _, version in (line.strip().partition("==") for line in f if line.strip())
    }

with open(requirements_path) as f:
    lines = f.readlines()

pinned_lines = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        pinned_lines.append(line)
        continue

    name = re.split(r"[=<>; \[]", stripped, maxsplit=1)[0]
    pinned_lines.append(f"{name}=={versions[normalise(name)]}\n")

with open(requirements_path, "w") as f:
    f.writelines(pinned_lines)
PYEOF
