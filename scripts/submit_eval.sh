#!/usr/bin/env bash

echo
echo ">>> Submitting eval jobs to slurm >>>"

###############################################################################
#                         hardware constraints                                #
#                                                                             #
# use & for AND, | for OR (e.g., "a100&icelake" or "a100|h200")              #
###############################################################################

GPU_CONSTRAINT="h200"            # e.g., h200, b200, a100, l40s — use | for OR
CPU_MEMORY="96G"                 # cpu memory allocation

###############################################################################
#                          evaluation configuration                           #
###############################################################################

# the models to evaluate (untrained or pretrained, see config/models.yaml)
models=(
    # "qwen2.5-coder-7b"
    # "qwen3-8b"
)

# comma-separated eval dataset names from config/eval_datasets.yaml, all evaluated
# within a single job per model - leave empty to evaluate every dataset in that file
datasets=""

# decoding profiles to evaluate with (from config/decoding.yaml)
decoding_profiles=(
    "greedy"
)

###############################################################################
#                            submit jobs                                      #
###############################################################################

# build datasets flag string
datasets_flag=""
if [ -n "$datasets" ]; then
    datasets_flag="-d $datasets"
fi

echo
echo "Evaluating ${#models[@]} models x ${#decoding_profiles[@]} decoding profiles"
echo "  Datasets: ${datasets:-all}"
echo "  Decoding profiles: ${decoding_profiles[*]}"
echo

for model in "${models[@]}"; do
    echo "Submitting jobs for $model"

    for decoding_profile in "${decoding_profiles[@]}"; do
        echo "  Submitting: [$decoding_profile]"

        sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=eval-${model}-${decoding_profile}
#SBATCH --output=logs/eval-${model}/${decoding_profile}-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --gpus=1
#SBATCH --mem=${CPU_MEMORY}
#SBATCH --constraint=${GPU_CONSTRAINT}

source ./scripts/setup_job.sh

# "command" forces bash to use the installed eval script, not its eval builtin
command eval \
    -m $model \
    $datasets_flag \
    --decoding-profile $decoding_profile

echo "Eval complete for model: $model, decoding: $decoding_profile"
EOF

        # small delay between submissions to avoid scheduler overload
        sleep 0.5
    done

    echo
done

echo "All jobs submitted!"
