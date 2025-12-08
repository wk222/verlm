#!/bin/bash
# ADPO Training - Qwen3-1.7B on WZX MATH Dataset
# Optimized for 2x A100 (80GB VRAM each) with Softmax loss

set -e  # Exit on error

echo "=========================================="
echo "ADPO Training - Qwen3 on WZX MATH (2x A100 80GB)"
echo "Loss Variant: Softmax"
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
export CUDA_VISIBLE_DEVICES=0,1  # 2 GPUs

# Configuration
CONFIG_NAME="adpo_qwen3_math_2a100"
OUTPUT_DIR="data/Qwen3-1.7B-ADPO-Softmax-2A100"
DATA_DIR="data/math_wzx"
N_GPUS=2

echo "Configuration:"
echo "  - Config: ${CONFIG_NAME}"
echo "  - Output: ${OUTPUT_DIR}"
echo "  - Data: ${DATA_DIR}"
echo "  - GPUs: ${CUDA_VISIBLE_DEVICES} (${N_GPUS} GPUs)"
echo "  - Loss: Softmax (Poly-Loss enabled)"
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
echo "🚀 Starting ADPO training with Softmax loss..."
echo ""

# ============================================================
# 2x A100 80GB 配置说明
# ============================================================
# - 使用 Softmax 损失变体 (原始 ADPO)
# - Poly-Loss 默认开启，增强训练信号
# - A100 80GB 显存充足，可以用更大的 batch size
# - 不需要 micro batch，直接处理完整 mini batch
# ============================================================

python -m verl.trainer.main_adpo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1280 \
    data.truncation=right \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.project_name="ADPO-pk-GRPO" \
    trainer.experiment_name=qwen3-1.7b-adpo-softmax-2a100 \
    wandb_config.project="ADPO-pk-GRPO" \
    wandb_config.name=qwen3-1.7b-adpo-softmax-2a100 \
    "$@"  # Pass any additional arguments

echo ""
echo "=========================================="
echo "✅ ADPO Training Complete!"
echo "=========================================="
