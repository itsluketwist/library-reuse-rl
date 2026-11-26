#!/usr/bin/env bash

echo
echo ">>> Submitting train jobs to slurm >>>"

###############################################################################
#                         cluster settings (isambard)                         #
###############################################################################

ACCOUNT="brics.u6yb"
PARTITION="workq"
GPUS=4                           # one node's gpus; the model is sharded across them (and
                                 # across nodes, for a model with "training_nodes" in
                                 # config/models.yaml)
TIME="24:00:00"                  # the workq qos limit; the job requeues and resumes if hit

# checkpoints are large and must outlive the run, so they go to project storage rather
# than $HOME (100GiB) or scratch (purged after 60 days of inactivity)
OUTPUT_BASE="${PROJECTDIR:?PROJECTDIR is not set - are you on isambard?}/$USER/library-reuse-rl"

###############################################################################
#                          rl training configuration                          #
###############################################################################

# the base models to train (see config/models.yaml), with their seed 42 runs (import, usage, target)
models=(
    # "qwen3-debug"
    "qwen3-8b"  # f2ca, 3079, f57b
    "olmo3-7b"  # e93f, cb85, 15e4
    "seed-coder-8b"  # 50e9, eee7, 5ad0
    # "qwen3-14b"  # on hold until there are hours to spare - trains across 2 nodes (see config/models.yaml), lu-usage only, as an ablation
)

# training jobs are dataset|reward-profile|prompt-style|learning-rate. keeping these together
# avoids silently applying a reward or a python-specific system prompt to the wrong task. the
# learning rate is optional - leave it off (e.g. "...|lu-v1|base") to use the recipe's.
#
# the chosen rewards (see the top of the library-usage section in config/rewards.yaml),
# trained for each model above for 4 epochs (num_train_epochs in the recipe). the ablations
# below ran on qwen3-8b only, with their run ids in brackets
training_jobs=(
    # the chosen rewards
    "data/train/library_usage|lu-import-matched|base|3e-6"
    "data/train/library_usage|lu-usage|base|3e-6"
    "data/train/library_usage|lu-target|base|3e-6"
    # done (qwen3-8b) - the shaping-weight ablations
    # "data/train/library_usage|lu-import|base|3e-6"  # (584b)
    # "data/train/library_usage|lu-import-s25|base|3e-6"  # (4db4)
    # "data/train/library_usage|lu-import-s0|base|3e-6"  # (9185)
    # "data/train/library_usage|lu-usage-s50|base|3e-6"  # (caf3)
    # "data/train/library_usage|lu-usage-s0|base|3e-6"  # (a342)
    # "data/train/library_usage|lu-target-s50|base|3e-6"  # (255a)
    # "data/train/library_usage|lu-target-s0|base|3e-6"  # (6626)
    # done - half credit for a library outside the target list
    # "data/train/library_usage|lu-target-half|base|3e-6"  # (8a38)
)

# the eval datasets (config/eval_datasets.yaml) each epoch checkpoint is scored on, always with
# the plain "base" prompt. every other eval setting (samples, decoding) comes from
# submit_eval.sh. the training
# data's own val and test splits are not here: training scores them every few steps itself
# (split_eval_steps in the recipe)
eval_datasets=(
    # LibReuseBench, test and control
    "libreusebench_test"
    "libreusebench_control"
    # regression tests of general coding ability - pass@1 shouldn't drop
    "humaneval_plus"
    "livecodebench"
)

# the eval datasets also scored with the prompt-engineering suffix ("appropriate", see
# TASK_SUFFIXES in src/prompts.py), at the same epochs, to compare training with prompting
suffix_eval_datasets=(
    "libreusebench_test"
    "libreusebench_control"
)

# which epoch checkpoints get those eval jobs: "all", or "final" for just the last epoch
eval_epochs="all"

# grpo training recipe - hyperparameters, including the decoding settings used to
# generate responses during rollout (see config/training.yaml)
recipe_config="config/training.yaml"

# accelerate config that shards the model across the job's gpus
accelerate_config="config/accelerate_configs/fsdp2.yaml"

# random seed for training - 42 for the first set of runs, 67 for the repeat
seed=67

# every epoch leaves a checkpoint (see save_strategy in the recipe); each is evaluated by
# a dependent job once training finishes. read from the recipe so the two cannot drift.
epochs=$(awk '/^num_train_epochs:/ {print $2}' "$recipe_config")

# true = tiny smoke test instead of full training (see --debug in src/run_grpo.py).
# for the larger debug runs, use scripts/submit_debug.sh instead.
debug=false

# RUN_ID (an environment variable) resumes an existing run instead of starting a new one,
# e.g. one whose job failed or was purged from slurm before it could be requeued:
#   RUN_ID=qwen3-8b-1f71 bash scripts/submit_grpo.sh
# training picks up from the run's newest complete checkpoint, and its epochs are evaluated
# as usual (epochs that already have results are skipped). the models and training_jobs
# above must then be just that run's own model and job

###############################################################################
#                            submit jobs                                      #
###############################################################################

# build debug flag string
debug_flag=""
if [ "$debug" = true ]; then
    debug_flag="--debug"
fi

# default to the tiny debug model if smoke-testing with no model set
if [ "$debug" = true ] && [ ${#models[@]} -eq 0 ]; then
    models=("qwen3-debug")
fi

# fail loudly instead of silently submitting nothing if the arrays above are empty
if [ ${#models[@]} -eq 0 ]; then
    echo "No models configured - uncomment/add at least one entry in the models array."
    exit 1
fi
if [ ${#training_jobs[@]} -eq 0 ]; then
    echo "No training jobs configured - add at least one entry to the training_jobs array."
    exit 1
fi
if [ -n "${RUN_ID:-}" ]; then
    if [ ${#models[@]} -ne 1 ] || [ ${#training_jobs[@]} -ne 1 ]; then
        echo "RUN_ID resumes one run, so set exactly one model and one training job (the run's own)."
        exit 1
    fi
    if [ ! -d "$OUTPUT_BASE/$RUN_ID" ]; then
        echo "No run to resume at $OUTPUT_BASE/$RUN_ID - check the run id."
        exit 1
    fi
    echo "Resuming existing run $RUN_ID from its newest checkpoint"
fi
for training_job in "${training_jobs[@]}"; do
    IFS="|" read -r training_dataset _reward_profile _prompt_style _learning_rate <<< "$training_job"
    if [ ! -f "$training_dataset/train.jsonl" ] || \
        [ ! -f "$training_dataset/val.jsonl" ]; then
        echo "Training dataset is missing train.jsonl or val.jsonl: $training_dataset"
        exit 1
    fi
done

echo
echo "Training ${#models[@]} models x ${#training_jobs[@]} dataset configurations"
echo "  Recipe: $recipe_config"
echo "  Sharding: $accelerate_config (${GPUS} gpus)"
echo "  Seed: $seed"
echo "  Epoch eval datasets: ${eval_datasets[*]} (epochs: $eval_epochs)"
echo "  ... and with the \"appropriate\" suffix: ${suffix_eval_datasets[*]}"
echo "  Checkpoints: $OUTPUT_BASE"

# one timestamped directory per invocation, holding the training logs and the logs of the
# eval jobs queued after them, so everything from one submission is in one place
run_timestamp=$(date +%y%m%dT%H%M%S)
log_dir="logs/${run_timestamp}"
mkdir -p "$log_dir"
echo "  Log directory: $log_dir"
echo

for model in "${models[@]}"; do
    # the nodes to train this model across: its "training_nodes" in config/models.yaml, or 1
    nodes=$(awk -v entry="$model:" '
        $0 == entry { found = 1; next }
        found && /^[^ #]/ { exit }
        found && $1 == "training_nodes:" { print $2; exit }
    ' config/models.yaml)
    nodes=${nodes:-1}
    echo "Submitting jobs for $model ($nodes node(s), $((GPUS * nodes)) gpus)"

    # one node launches accelerate directly. several each run a copy of it under srun, and
    # accelerate joins them into one run with the first node as the meeting point (the
    # escaped variables are filled in when the job runs, on its own nodes)
    launch="\$HOST_MEMBIND accelerate launch"
    if [ "$nodes" -gt 1 ]; then
        launch="srun --nodes=$nodes --ntasks-per-node=1 --cpus-per-task=\$SLURM_CPUS_ON_NODE \
--gpus-per-node=$GPUS --kill-on-bad-exit=1 \$HOST_MEMBIND bash scripts/launch_on_node.sh \
--num_machines $nodes --main_process_port \$MASTER_PORT \
--main_process_ip \$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)"
    fi

    for training_job in "${training_jobs[@]}"; do
        IFS="|" read -r training_dataset reward_profile library_prompt_style learning_rate \
            <<< "$training_job"
        dataset_name=$(basename "$training_dataset")

        # the optional learning rate overrides the recipe's, and is named in the job and log
        learning_rate_flag=""
        job_name="train-${model}-${dataset_name}-${library_prompt_style}"
        if [ -n "$learning_rate" ]; then
            learning_rate_flag="--learning-rate $learning_rate"
            job_name="${job_name}-lr${learning_rate}"
        fi

        # prefix the epoch checkpoints are registered under (config/trained_models.yaml), fixed
        # here so the dependent eval jobs below can name them before training finishes
        run_id="${RUN_ID:-${model}-$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-4)}"

        echo "  Submitting: $dataset_name [$reward_profile] [$library_prompt_style] [lr ${learning_rate:-recipe}]"

        # the time limit in seconds (TIME is hours:minutes:seconds), so the job can requeue
        # itself at an epoch's end when the next epoch wouldn't fit (see the loop below)
        IFS=: read -r limit_hours limit_minutes limit_seconds <<< "$TIME"
        time_limit_seconds=$((10#$limit_hours * 3600 + 10#$limit_minutes * 60 + 10#$limit_seconds))

        train_job_id=$(sbatch --parsable <<EOF
#!/bin/bash -l
#SBATCH --job-name=reuse-${job_name}
#SBATCH --output=${log_dir}/${job_name}-%j.out
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${nodes}
#SBATCH --gpus=$((GPUS * nodes))
#SBATCH --time=${TIME}
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

source ./scripts/setup_job.sh

# requeue on the time-limit warning signal - training resumes from its newest checkpoint (one
# every resume_save_steps steps, see the recipe), so at most those steps are redone
trap 'echo "time limit approaching - requeuing \$SLURM_JOB_ID"; scontrol requeue \$SLURM_JOB_ID; exit 0' USR1

# training ends after every checkpoint save (restart_after_save in the recipe: a save
# leaves gpu memory that only a fresh process gets back), so it is relaunched here, resuming
# from that checkpoint, until it writes training_complete (TRAINING_COMPLETE_FILE in
# src/grpo/train_model.py) or fails. relaunching on this node skips the queue
job_start=\$(date +%s)
launches=0
while true; do
    launches=\$((launches + 1))
    launch_start=\$(date +%s)
    # --module runs src.run_grpo as "python -m" would; the flags after it are the script's.
    # \$HOST_MEMBIND (see setup_job.sh) stops cpu-side memory spilling into the gpus
    $launch --config_file $accelerate_config --num_processes $((GPUS * nodes)) \
        --module src.run_grpo \
        --model $model \
        -d $training_dataset \
        --recipe-config $recipe_config \
        -s $seed \
        --library-prompt-style $library_prompt_style \
        --output-root ${OUTPUT_BASE} \
        --run-id $run_id \
        --reward-profile $reward_profile \
        $learning_rate_flag \
        $debug_flag &
    wait \$!
    # keep training's exit status, so a failed run shows as FAILED in slurm and its eval
    # jobs (held with afterok) never start. a requeue exits through the trap above instead
    status=\$?
    if [ \$status -ne 0 ] || [ -f "${OUTPUT_BASE}/${run_id}/training_complete" ]; then
        break
    fi
    # a guard against relaunching forever, if training ever stopped without progressing
    if [ \$launches -ge 50 ]; then
        echo "Launched training \$launches times without it finishing - giving up"
        status=1
        break
    fi
    # requeue now, at the epoch boundary, if another launch (an epoch) won't finish in the
    # job's time left - estimated from the last launch, plus 10% - rather than lose a
    # partly trained epoch to the time limit
    launch_seconds=\$((\$(date +%s) - launch_start))
    seconds_left=\$(($time_limit_seconds - (\$(date +%s) - job_start)))
    if [ \$seconds_left -lt \$((launch_seconds * 11 / 10)) ]; then
        echo "The next epoch won't fit in the \$((seconds_left / 60)) minutes left - requeuing [\$(date '+%Y-%m-%d %H:%M:%S')]"
        scontrol requeue \$SLURM_JOB_ID
        exit 0
    fi
    echo "Relaunching training from its newest checkpoint [\$(date '+%Y-%m-%d %H:%M:%S')]"
done

echo "RL training finished for model: $model, dataset: $dataset_name, exit status \$status [\$(date '+%Y-%m-%d %H:%M:%S')]"
exit \$status
EOF
)

        echo "  Submitted train job $train_job_id (run id: $run_id) - follow it with:"
        echo "    tail -f ${log_dir}/${job_name}-${train_job_id}.out"

        # queue one single-gpu eval job per epoch checkpoint, held until training succeeds.
        # submit_eval.sh owns the other eval settings (samples, decoding, prompt style), so
        # they are configured in one place only; the datasets are eval_datasets above, and
        # suffix_eval_datasets again with the "appropriate" suffix.
        if [ "$debug" != true ]; then
            first_epoch=1
            if [ "$eval_epochs" = "final" ]; then
                first_epoch=$epochs
            fi
            eval_models=""
            for epoch in $(seq "$first_epoch" "$epochs"); do
                eval_models="$eval_models vllm:${run_id}-epoch${epoch}"
            done
            echo "  Queueing eval jobs for epochs $first_epoch-$epochs (after $train_job_id)"
            EVAL_MODELS="$eval_models" \
                EVAL_DATASETS="${eval_datasets[*]}" \
                EVAL_PROMPT_STYLES="base" \
                EVAL_DEPENDENCY="afterok:${train_job_id}" \
                EVAL_LOG_DIR="$log_dir" \
                bash scripts/submit_eval.sh
            EVAL_MODELS="$eval_models" \
                EVAL_DATASETS="${suffix_eval_datasets[*]}" \
                EVAL_PROMPT_STYLES="appropriate" \
                EVAL_DEPENDENCY="afterok:${train_job_id}" \
                EVAL_LOG_DIR="$log_dir" \
                bash scripts/submit_eval.sh
        fi

        # small delay between submissions to avoid scheduler overload
        sleep 0.5
    done

    echo
done

echo "All jobs submitted!"
