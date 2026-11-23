#!/usr/bin/env bash
# runs accelerate launch on one node of a multi-node training job. scripts/submit_grpo.sh
# starts one copy per node with srun, and each passes on its own node's rank (from slurm),
# so the nodes' processes join up as one run. the other arguments are accelerate launch's.

exec accelerate launch --machine_rank "$SLURM_NODEID" "$@"
