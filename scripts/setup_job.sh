#!/usr/bin/env bash

echo "============================================================"
echo "Setting up environment!"
echo "============================================================"
echo

echo "Running on node: $(hostname)"
echo "Date/time: $(date)"
echo
echo

# display cpu and memory info
echo "CPU model: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "CPU cores: $(nproc)"
echo "Total memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Available memory: $(free -h | awk '/^Mem:/ {print $7}')"
echo

echo "Loading virtual environment..."
source .venv/bin/activate
echo

echo "Loading environment variables from .env..."
if [ -f .env ]; then
    # `set -a` marks every variable sourcing sets as exported, so run_eval and any other
    # child process actually sees them via os.environ - a plain `source .env` only sets
    # shell-local variables unless the file itself uses `export`, which ours doesn't.
    set -a
    source .env
    set +a
else
    echo "WARNING: .env file not found! Create one with HF_HOME, HF_TOKEN, etc."
fi
echo

# reduce cuda allocator fragmentation (helps avoid oom on long sequences)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# unique port per job, so co-located jobs don't race for the same vllm port
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 10000))

echo "Environment setup, checking CUDA versions..."

# api-provider eval jobs (config/models.yaml "provider: api") run on cpu-partition nodes
# with no GPU driver at all (see scripts/submit_eval.sh) - skip the GPU checks there
# rather than failing the job on a command that can't exist.
if command -v nvidia-smi &> /dev/null; then
    nvidia-debugdump -l
    cat /proc/driver/nvidia/version
    echo
    echo "CUDA version:"
    nvcc --version

    nvidia-smi
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    echo

    echo "Now test python CUDA access..."

    test_cuda  # from llm_cgr

    echo "CUDA tested, starting job..."
else
    echo "No GPU driver on this node - skipping CUDA checks."
fi

echo
echo
