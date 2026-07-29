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
    source .env
else
    echo "WARNING: .env file not found! Create one with HF_HOME, HF_TOKEN, etc."
fi
echo

# reduce cuda allocator fragmentation (helps avoid oom on long sequences)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# unique port per job, so co-located jobs don't race for the same vllm port
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 10000))

echo "Environment setup, checking CUDA versions..."

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

echo
echo
