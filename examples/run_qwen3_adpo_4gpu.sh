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
# 目标显存占用: 18-20GB (75-80%)
# ============================================================
# 配置说明:
# - train_batch_size=128, ppo_mini_batch_size=64
# - ppo_micro_batch_size_per_gpu=8: 每 GPU 8 个样本
# - log_prob_micro_batch_size_per_gpu=8: log prob 批次
# - gpu_memory_utilization=0.45: rollout 显存占用
# - rollout.n=8: 每提示生成 8 个响应
#
# ============================================================
# Batch Size 约束条件 (调整时必须满足):
# ============================================================
# 1. normalized_ppo_mini_batch_size = ppo_mini_batch_size * rollout.n / n_gpus
#    当前: 32 * 8 / 4 = 64
#
# 2. 约束: normalized_ppo_mini_batch_size % ppo_micro_batch_size_per_gpu == 0
#    当前: 64 % 8 == 0 ✓
#    可选值: 1, 2, 4, 8, 16, 32, 64 (64的因子)
#
# 3. log_prob_micro_batch_size_per_gpu 无严格整除约束，可自由调整
#
# 显存优化建议:
# - 降低 ppo_micro_batch_size_per_gpu: 4 或 2 (减少训练显存峰值)
# - 降低 log_prob_micro_batch_size_per_gpu: 4 或 2 (减少log prob计算显存)
# ============================================================

python -m verl.trainer.main_adpo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=880 \
    data.max_response_length=1280 \
    data.truncation=right \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_seqs=192 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.model.enable_activation_offload=True \
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
