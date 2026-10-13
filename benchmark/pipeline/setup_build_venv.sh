#!/usr/bin/env bash
# create the throwaway environment used only by the benchmark construction pipeline.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BUILD_VENV=".venv-build"

# generated code only ever runs here, never in the project's own venv. stages 02 and 03
# temporarily install each library solution's packages, so seed the environment with pip.
uv venv "$BUILD_VENV" --python 3.12.12 --seed --clear

echo
echo "build venv ready at $BUILD_VENV"
echo "the build stages use it by default, e.g.:"
echo "  python benchmark/pipeline/02_generate.py --limit 1"
