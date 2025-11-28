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

# Batch Size Calculation for 4x4090:
# - Rollout Batch Size (per GPU): 8
# - Num Generations (n): 8
# - Total Rollout per Step: 4 GPUs * 8 prompts * 8 gens = 256 sequences
# - Mini-batch Size: 64 (Update batch size)
# - Micro-batch Size: 4 (Gradient accumulation per device)
# - Gradient Accumulation Steps: 64 / (4 GPUs * 4 micro) = 4

python -m verl.trainer.main_adpo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
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
