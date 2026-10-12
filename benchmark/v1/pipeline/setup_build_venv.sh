#!/usr/bin/env bash
# create the throwaway environment used only by the benchmark construction pipeline.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BUILD_VENV=".venv-build"

# stage 04 needs only the standard library; stage 05 temporarily installs each
# library solution's declared packages, so seed the environment with pip.
uv venv "$BUILD_VENV" --python 3.12.12 --seed --clear

echo
echo "build venv ready at $BUILD_VENV"
echo "run the build stages with:"
echo "  LRB_BUILD_PYTHON=\"\$(pwd)/$BUILD_VENV/bin/python\" python benchmark/pipeline/04_make_tests.py"
