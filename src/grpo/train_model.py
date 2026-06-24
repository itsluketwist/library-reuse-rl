# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The GRPO training loop, wired to this project's reward and dataset format."""

import logging
import os
import sys
from pathlib import Path

import datasets
import transformers
from transformers import TrainerCallback, set_seed
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

from src.data import load_train_dataset
from src.utils.config import load_full_yaml

from .model_utils import GRPOConfig, get_model, get_tokenizer
from .rewards import RewardBit, make_library_reuse_reward
from .wandb_logging import init_wandb_training


logger = logging.getLogger(__name__)

# sample counts used by --debug mode (src/run_grpo.py) - just enough to fill a train
# batch and give eval a couple of rows (see config/train/demo.yaml), small enough to
# finish in seconds. GRPOTrainer still takes a real optimizer step on a partial/short
# epoch like this - transformers' Trainer flushes any accumulated gradients at epoch end.
DEBUG_TRAIN_SAMPLES = 16
DEBUG_VAL_SAMPLES = 4


class SaveTrainerStateCallback(TrainerCallback):
    """
    Periodically overwrites output_dir/trainer_state.json with the training history so
    far, on the same cadence as checkpointing (`save_steps`) - so if a run crashes, the
    metric history up to the last save isn't lost along with it.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.is_world_process_zero and state.global_step % args.save_steps == 0:
            state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))


def train_model_with_grpo(
    model_name_or_path: str,
    dataset_dir: str | Path,
    output_dir: str | Path,
    reward_bits: RewardBit,
    seed: int,
    recipe_config: str = "config/train/demo.yaml",
    debug: bool = False,
) -> None:
    """
    Run GRPO training for `model_name_or_path` against the skywork-format dataset at
    `dataset_dir` (see src/data.py::load_train_dataset), saving the trained model to
    `output_dir`.

    `seed` overrides the corresponding field parsed from `recipe_config`, so that -
    along with the model, dataset, and output path - it's resolved once by the caller
    (src/run_grpo.py) rather than duplicated in the recipe file. Every other training
    hyperparameter, including `system_prompt` (used to build the dataset's prompts)
    and the decoding settings used during rollout generation (temperature/top_p/top_k),
    comes straight from `recipe_config`.

    If `debug` is set, the train/val splits are truncated to DEBUG_TRAIN_SAMPLES /
    DEBUG_VAL_SAMPLES rows, to quickly smoke-test that training runs end-to-end.
    """
    training_args, model_args = TrlParser((GRPOConfig, ModelConfig)).parse_dict(
        load_full_yaml(recipe_config)
    )

    model_args.model_name_or_path = model_name_or_path
    training_args.output_dir = str(output_dir)
    training_args.seed = seed

    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ###############
    # Load dataset
    ###############
    dataset = load_train_dataset(
        dataset_dir=dataset_dir,
        system_prompt=training_args.system_prompt,
        train_n=DEBUG_TRAIN_SAMPLES if debug else None,
        val_n=DEBUG_VAL_SAMPLES if debug else None,
    )

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get the reward function, configured with this run's reward_bits
    reward_func = make_library_reuse_reward(reward_bits)

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
        callbacks=[SaveTrainerStateCallback()],
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {"tags": ["library-reuse-rl"]}
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset["val"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
