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
DATA_DIR="data/math_level3"  # HF: wzx111/MATH-lighteval-level3
N_GPUS=4

echo "Configuration:"
echo "  - Config: ${CONFIG_NAME}"
echo "  - Output: ${OUTPUT_DIR}"
echo "  - Data: ${DATA_DIR}"
echo "  - GPUs: ${CUDA_VISIBLE_DEVICES} (${N_GPUS} GPUs)"
echo ""

# Download HF Level 3 dataset if missing (requires HF token if private)
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "📥 Downloading HF dataset wzx111/MATH-lighteval-level3 -> ${DATA_DIR}/train.parquet"
    mkdir -p ${DATA_DIR}
    python3 -c "
from datasets import load_dataset
import os

ds = load_dataset('wzx111/MATH-lighteval-level3', split='train')
save_path = os.path.join('${DATA_DIR}', 'train.parquet')
ds.to_parquet(save_path)
print(f'Saved {len(ds)} rows to {save_path}')
"
    echo ""
else
    echo "✅ HF Level3 dataset already exists at ${DATA_DIR}"
    echo ""
fi

# Run ADPO training
echo "🚀 Starting ADPO training..."
echo ""

# ============================================================
# Decoupled Loss 配置 - num_generations=8
# ============================================================
# 当前使用 Decoupled Loss 变体：
# - loss_variant: decoupled
# - tau: 0.5
# - beta_reward: 0.3
# - num_generations: 8
#
# 关键参数解耦说明：
# - ppo_micro_batch_size_per_gpu=6: 每个GPU处理的样本数
# - num_generations=8: 每个prompt生成的response数
# - 这两个参数是独立的！
# ============================================================

python -m verl.trainer.main_adpo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1280 \
    data.truncation=left \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.project_name="ADPO-pk-GRPO" \
    trainer.experiment_name=qwen3-1.7b-adpo-decoupled-8gen \
    wandb_config.project="ADPO-pk-GRPO" \
    wandb_config.name=qwen3-1.7b-adpo-decoupled-8gen \
    "$@"  # Pass any additional arguments

echo ""
echo "=========================================="
echo "✅ ADPO Training Complete!"
echo "=========================================="
