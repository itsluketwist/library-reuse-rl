#!/usr/bin/env bash
# sourced at the top of every slurm job: activates the venv, loads .env, and points the
# large/churny paths at isambard's storage areas (see the storage table in the README).

echo "============================================================"
echo "Setting up environment!"
echo "============================================================"
echo

echo "Running on node: $(hostname)"
echo "Date/time: $(date)"
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

# $HOME is only 100GiB and the model cache runs to hundreds of gigabytes, so fail here
# rather than filling the quota mid-download.
if [ -z "$HF_HOME" ]; then
    echo "ERROR: HF_HOME is not set - point it at \$PROJECTDIR in .env (see .env.example)."
    return 1 2>/dev/null || exit 1
fi
case "$HF_HOME" in
    "$HOME"/*)
        echo "ERROR: HF_HOME is inside \$HOME (100GiB quota): $HF_HOME"
        echo "Point it at \$PROJECTDIR in .env instead (see .env.example)."
        return 1 2>/dev/null || exit 1
        ;;
esac
echo "Model cache (HF_HOME): $HF_HOME"

# compile caches, wandb staging and the disposable evaluation venvs are all regenerable
# churn, which is what scratch is for. keeping them between jobs saves recompiling.
if [ -n "$SCRATCHDIR" ]; then
    export TORCHINDUCTOR_CACHE_DIR="$SCRATCHDIR/cache/torchinductor"
    export TRITON_CACHE_DIR="$SCRATCHDIR/cache/triton"
    export VLLM_CACHE_ROOT="$SCRATCHDIR/cache/vllm"
    export WANDB_DIR="$SCRATCHDIR/wandb"
    # read by src/run_eval.py, overriding environment_root in config/eval_datasets.yaml
    export EVAL_ENVIRONMENT_ROOT="$SCRATCHDIR/libreuseeval-venvs"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" \
        "$WANDB_DIR" "$EVAL_ENVIRONMENT_ROOT"
    echo "Caches and eval venvs: $SCRATCHDIR"
else
    echo "WARNING: SCRATCHDIR is not set - caches and eval venvs stay in \$HOME."
fi
echo

# reduce cuda allocator fragmentation (helps avoid oom on long sequences)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# deepgemm (vllm's fp8 kernels) can't load here as the nodes have no cuda toolkit, and
# vllm's startup check prints a long traceback about it. we run bf16 so never need it.
export VLLM_USE_DEEP_GEMM=0

# unique port per job, so co-located jobs don't race for the same vllm port
export MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 10000))

# a job across several nodes (a model too large for one, see "training_nodes" in
# config/models.yaml) sends nccl's traffic between the nodes over isambard's slingshot
# network, through the aws-ofi-nccl plugin (paths from `module show brics/aws-ofi-nccl/1.8.1`,
# settings from isambard's own multi-node setup, brics/apptainer-multi-node). the module
# itself isn't loaded, as it would also put an older nccl ahead of torch's own
if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
    ofi_nccl_lib="/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/aws-ofi-nccl-1.8.1-c47cd5ivrugm3jzlyqyis4igyflnydmo/lib"
    export LD_LIBRARY_PATH="$ofi_nccl_lib:/opt/cray/libfabric/2.3.1/lib64:${LD_LIBRARY_PATH:-}"
    # fail if the plugin doesn't load, rather than fall back to (far slower) tcp sockets
    export NCCL_NET="AWS Libfabric"
    export NCCL_SOCKET_IFNAME="hsn"
    export NCCL_CROSS_NIC="0"
    export NCCL_NET_GDR_LEVEL="PHB"
    export NCCL_MIN_NCHANNELS="4"
    export NCCL_GDRCOPY_ENABLE="1"
    export NCCL_NET_FORCE_FLUSH="1"
    export FI_MR_CACHE_MONITOR="userfaultfd"
    export FI_CXI_DISABLE_HOST_REGISTER="1"
    export FI_HMEM_CUDA_USE_GDRCOPY="1"
    export FI_CXI_DEFAULT_CQ_SIZE="131072"
    export FI_CXI_DEFAULT_TX_SIZE="1024"
    export FI_CXI_RDZV_PROTO="alt_read"
    export FI_CXI_RDZV_THRESHOLD="0"
    export FI_CXI_RDZV_GET_MIN="0"
    export FI_CXI_RDZV_EAGER_SIZE="0"
    echo "Multi-node job: ${SLURM_JOB_NUM_NODES} nodes, nccl over slingshot (aws-ofi-nccl)"
fi

# on isambard's gh200 nodes each gpu's memory is also visible to linux as a numa node, so
# ordinary cpu-side memory (e.g. model weights being loaded) can silently fill a gpu. the
# training launch is prefixed with $HOST_MEMBIND, which keeps cpu-side memory on the
# numa nodes that have cpus (their own lpddr memory) - gpu allocations are unaffected.
HOST_MEMBIND=""
if command -v numactl &> /dev/null; then
    cpu_numa_nodes=$(numactl --hardware | awk '/^node [0-9]+ cpus: [0-9]/ {print $2}' | paste -sd, -)
    if [ -n "$cpu_numa_nodes" ]; then
        HOST_MEMBIND="numactl --membind=${cpu_numa_nodes}"
    fi
fi
export HOST_MEMBIND
echo "Host memory binding: ${HOST_MEMBIND:-none (numactl unavailable)}"

# api-provider eval jobs run without a gpu, so skip the gpu report there rather than
# failing the job on a command that cannot exist.
if command -v nvidia-smi &> /dev/null; then
    echo "GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
    # the cuda version torch was built for is what actually matters at runtime - see
    # scripts/check_gpu_stack.py for the full check of the installed stack
    python -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda)"
else
    echo "No GPU driver on this node - skipping GPU checks."
fi

echo
echo
