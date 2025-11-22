#!/bin/bash
# ADPO Training Example - GSM8K Dataset
# This script demonstrates how to train a model using ADPO on the GSM8K math reasoning dataset

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Basic ADPO training with on-policy anchor (like GRPO)
python -m verl.trainer.main_adpo \
    --config-name adpo_trainer \
    algorithm.adv_estimator=adpo \
    algorithm.num_generations=8 \
    algorithm.tau=0.8 \
    algorithm.anchor_update_mode=on_policy \
    algorithm.use_adaptive_tau=True \
    algorithm.adaptive_tau_alpha=0.5 \
    algorithm.beta_reward=0.5 \
    algorithm.drop_all_failed_prompts=False \
    trainer.project_name=adpo_gsm8k \
    trainer.experiment_name=adpo_on_policy \
    trainer.total_epochs=30 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.train_files=[data/gsm8k/train.parquet] \
    data.val_files=[data/gsm8k/val.parquet] \
    actor_rollout_ref.model.path=Qwen/Qwen2-0.5B-Instruct

echo "ADPO training completed!"

