#!/bin/bash
# Reproduce Open-R1 ADPO Baseline - Qwen3-1.7B on WZX MATH Dataset
# Optimized for 4x4090 (24GB VRAM each)

set -e  # Exit on error

echo "=========================================="
echo "ADPO Reproduction - Qwen3 on WZX MATH (4x4090)"
echo "=========================================="
echo ""

# Check if we're in the verlm directory
if [ ! -d "verl/trainer/adpo" ]; then
    echo "❌ Error: Please run this script from the verlm/ directory."
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 4 GPUs

# Configuration
CONFIG_NAME="adpo_qwen3_math_hybrid" # Use our new hybrid config
OUTPUT_DIR="data/Qwen3-1.7B-Open-R1-ADPO-WZX"
DATA_DIR="data/math_wzx"
N_GPUS=4

echo "Configuration:"
echo "  - Config: ${CONFIG_NAME}"
echo "  - Output: ${OUTPUT_DIR}"
echo "  - Data: ${DATA_DIR}"
echo "  - GPUs: ${CUDA_VISIBLE_DEVICES} (${N_GPUS} GPUs)"
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

# Run ADPO training
echo "🚀 Starting ADPO training..."
echo ""

# Batch Size Calculation for 4x4090 (24GB VRAM each):
# ============================================================
# 约束: train_batch_size >= ppo_mini_batch_size
# ============================================================
# - train_batch_size: 每个训练步骤的提示数量
# - ppo_mini_batch_size: PPO 更新的 mini-batch 大小 (必须 <= train_batch_size)
# - ppo_micro_batch_size_per_gpu: 每 GPU 的微批次大小 (用于梯度累积)
# - rollout.n: 每个提示生成的响应数量
#
# 配置说明:
# - train_batch_size=128: 每步使用 128 个提示
# - ppo_mini_batch_size=64: PPO 每次更新用 64 个样本
# - ppo_micro_batch_size_per_gpu=4: 每 GPU 处理 4 个样本 (梯度累积 = 64/(4*4)=4)
# - rollout.n=4: 每提示生成 4 个响应 (总序列数 = 128*4=512)
# - max_prompt_length=1024: 支持长 prompt (有些数据 800+ tokens)
# - max_response_length=1280: 支持 1200+ 的 response
# - truncation=left: 超长 prompt 从左边截断 (保留最近内容)

python -m verl.trainer.main_adpo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1280 \
    data.truncation=left \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.experiment_name=qwen3-1.7b-adpo-wzx-4gpu \
    wandb_config.name=qwen3-1.7b-adpo-wzx-4gpu \
    "$@"  # Pass any additional arguments

echo ""
echo "=========================================="
echo "✅ ADPO Training Complete!"
echo "=========================================="
