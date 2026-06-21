#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxx"
export HF_HOME=xxxxxxxxx
export HF_LOCAL_HOME=xxxxxxxxxx
export HF_DATASETS_CACHE=xxxxxxxxxxxx
export TRITON_CACHE_DIR=/tmp/triton
export CUDA_VISIBLE_DEVICES=0,1 # change based on the gpu number


export E2B_API_KEY=e2b_xxxxxxxxxxxxx # need to apply for E2B API key

export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxx" # need to apply for E2B API key

wandb login $WANDB_API_KEY

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes 1 --main_process_port 29501 \
    src/open_r1/grpo.py --config recipes/config_demo.yaml
