#!/bin/bash
# GSPO Training (Aligned) - Qwen3 on 4x4090
# Fair Comparison with ADPO (Softmax/Decoupled)
#
# Aligned Parameters:
# - Dataset: math_level3 (Same as ADPO experiments)
# - Batch Sizes: train=120, mini=60 (Same as ADPO)
# - Rollout: n=6 (Same as ADPO)
# - Length: 1024 (Same as ADPO)

set -e

echo "=========================================="
echo "GSPO Aligned Training - Qwen3 (4x4090)"
echo "=========================================="

if [ ! -d "verl/trainer" ]; then
    echo "❌ Error: Please run from verlm/ directory."
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
CONFIG_NAME="gspo_qwen3_math_hybrid"
OUTPUT_DIR="data/Qwen3-1.7B-GSPO-Aligned-WZX"
DATA_DIR="data/math_level3"  # Align dataset with ADPO
N_GPUS=4

echo "Config: ${CONFIG_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Data:   ${DATA_DIR}"

# Ensure dataset exists (Use math_level3_dataset.py to match ADPO)
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "📥 Preprocessing MATH Level 3 dataset..."
    python3 examples/data_preprocess/math_level3_dataset.py \
        --local_save_dir ${DATA_DIR}
fi

echo "🚀 Starting GSPO training..."

# Overrides to match ADPO configuration exactly
python -m verl.trainer.main_ppo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    data.train_batch_size=120 \
    data.val_batch_size=24 \
    data.max_prompt_length=1024 \
    data.max_response_length=1280 \
    data.truncation=left \
    data.shuffle=True \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_seqs=120 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=60 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.n_gpus_per_node=${N_GPUS} \
    +trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.project_name="ADPO-pk-GRPO" \
    trainer.experiment_name=qwen3-1.7b-gspo-aligned-4gpu \
    wandb_config.project="ADPO-pk-GRPO" \
    wandb_config.name=qwen3-1.7b-gspo-aligned-4gpu \
    "$@"

echo "✅ GSPO Aligned Training Complete!"

