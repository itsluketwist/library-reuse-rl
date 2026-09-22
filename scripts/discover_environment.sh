#!/usr/bin/env bash
# report available cluster software and resources without installing or loading anything.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: bash scripts/discover_environment.sh [report.txt]"
    echo "Run on the login node and again inside a GPU allocation. No project environment is required."
    exit 0
fi
if [[ $# -gt 1 ]]; then
    echo "Expected at most one argument: the output report path." >&2
    exit 2
fi

report=${1:-environment-report.txt}
mkdir -p "$(dirname "$report")"
: > "$report"

run() {
    # absent tools and unavailable scheduler permissions are useful discovery results.
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Not available in the current environment."
    elif [[ $(type -t "$1") == file ]] && command -v timeout >/dev/null 2>&1; then
        timeout 20 "$@" || echo "Command exited with status $? (124 means timeout)."
    else
        "$@" || echo "Command exited with status $?."
    fi
}

collect_report() {
    printf 'Environment discovery: %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf 'Host: %s\nWorking directory: %s\n' "$(hostname)" "$PWD"
    printf 'Slurm job: %s\n' "${SLURM_JOB_ID:-none; probably a login node}"

    printf '\n=== system and storage ===\n'
    run uname -a
    run lscpu
    run free -h
    run df -h .
    run lfs quota -u "$(id -un)" "$PWD"

    printf '\n=== loaded and available modules ===\n'
    # module is normally a shell function inherited from the site's login environment.
    run module -t list
    run module -t avail

    printf '\n=== gpu, driver, and current cuda toolkit ===\n'
    run nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    run nvidia-smi topo -m
    run nvcc --version

    printf '\n=== python and installed packages ===\n'
    run python3 --version
    run python --version
    discovery_python=""
    if command -v python3 >/dev/null 2>&1; then
        discovery_python=python3
    elif command -v python >/dev/null 2>&1; then
        discovery_python=python
    fi
    if [[ -n "$discovery_python" ]]; then
        # reading package metadata does not import torch, initialize cuda, or require a venv.
        run "$discovery_python" - <<'PY'
import platform
import sys

print("Executable:", sys.executable)
print("Python:", sys.version)
print("Architecture:", platform.machine())
print("libc:", platform.libc_ver())
try:
    from importlib import metadata
except ImportError:
    print("Package metadata requires Python 3.8 or newer.")
else:
    for name in (
        "pip", "torch", "transformers", "trl", "accelerate", "deepspeed",
        "vllm", "unsloth", "flash-attn", "flash-linear-attention", "causal-conv1d",
    ):
        try:
            print(name + ": " + metadata.version(name))
        except metadata.PackageNotFoundError:
            print(name + ": not installed in this Python")
PY
    fi

    printf '\n=== build and container tools ===\n'
    run gcc --version
    run cmake --version
    run uv --version
    run apptainer --version
    run podman --version

    printf '\n=== scheduler and account limits ===\n'
    run sinfo -o '%P %l %G %c %m'
    run scontrol show partition
    run sacctmgr -nP show assoc "user=$(id -un)" format=Account,Partition,QOS,GrpTRESMins,MaxWall

    printf '\nReport saved to %s\n' "$report"
    echo "Missing GPU tools on a login node do not describe the compute-node environment."
    echo "Repeat inside a GPU allocation; also confirm project storage quotas and allocation balance."
}

collect_report 2>&1 | tee -a "$report"
