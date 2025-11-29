#!/bin/bash
# DAPO Training - Qwen3-1.7B on WZX MATH Dataset
# DAPO = Dynamic Advantage Preference Optimization
# 特点: 非对称裁剪 (clip_ratio_low=0, clip_ratio_high=0.28)
# Optimized for 4x4090 (24GB VRAM each)

set -e

echo "=========================================="
echo "DAPO Training - Qwen3 on 4x4090"
echo "=========================================="
echo ""

# Check if we're in the verlm directory
if [ ! -d "verl/trainer" ]; then
    echo "❌ Error: Please run this script from the verlm/ directory."
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
CONFIG_NAME="dapo_qwen3_math_hybrid"
OUTPUT_DIR="data/Qwen3-1.7B-DAPO-WZX"
DATA_DIR="data/math_wzx"
N_GPUS=4

echo "Configuration:"
echo "  - Config: ${CONFIG_NAME}"
echo "  - Output: ${OUTPUT_DIR}"
echo "  - Data: ${DATA_DIR}"
echo "  - GPUs: ${CUDA_VISIBLE_DEVICES} (${N_GPUS} GPUs)"
echo "  - Algorithm: DAPO (Asymmetric Clipping GRPO)"
echo ""

# Download and preprocess WZX MATH dataset if not exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "📥 Downloading and preprocessing WZX MATH dataset..."
    python3 examples/data_preprocess/math_wzx_dataset.py \
        --local_save_dir ${DATA_DIR}
    echo ""
else
    echo "✅ WZX MATH dataset already exists at ${DATA_DIR}"
    echo ""
fi

echo "🚀 Starting DAPO training..."
echo ""

# ============================================================
# DAPO 配置说明:
# ============================================================
# - adv_estimator: grpo (使用GRPO的advantage估计)
# - policy_loss.loss_mode: vanilla (标准PPO损失)
# - clip_ratio_low: 0.0 (DAPO核心: 移除下界裁剪)
# - clip_ratio_high: 0.28 (非对称上界裁剪)
# - norm_adv_by_std_in_grpo: False (不使用标准差归一化，Dr.GRPO风格)
# - 其他配置与ADPO保持一致以保证公平对比
# ============================================================

python -m verl.trainer.main_ppo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=880 \
    data.max_response_length=1280 \
    data.truncation=right \
    data.shuffle=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_seqs=300 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio_low=0.0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.project_name="ADPO-pk-GRPO" \
    trainer.experiment_name=qwen3-1.7b-dapo-wzx-4gpu \
    wandb_config.project="ADPO-pk-GRPO" \
    wandb_config.name=qwen3-1.7b-dapo-wzx-4gpu \
    "$@"

echo ""
echo "=========================================="
echo "✅ DAPO Training Complete!"
echo "=========================================="
