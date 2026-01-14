#!/bin/bash
# GSPO Training - Qwen3-1.7B on WZX MATH Dataset
# Optimized for 2x A100 (80GB VRAM each)
# Fair comparison with ADPO

set -e  # Exit on error

echo "=========================================="
echo "GSPO Training - Qwen3 on WZX MATH (2x A100 80GB)"
echo "Algorithm: GSPO (GRPO + Sequence-level)"
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
export CUDA_VISIBLE_DEVICES=0,1  # 2 GPUs
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Configuration
CONFIG_NAME="gspo_qwen3_math_2a100"
OUTPUT_DIR="data/Qwen3-1.7B-GSPO-2A100"
DATA_DIR="data/math_wzx"
N_GPUS=2

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

# Run GSPO training
echo "🚀 Starting GSPO training..."
echo ""

# ============================================================
# 2x A100 80GB 配置说明 (与 ADPO 对齐)
# ============================================================
# - 使用 GSPO (GRPO_SEQ)
# - Batch Size = 64 (Global)
# - N = 16 (Generations)
# ============================================================

python -m verl.trainer.main_ppo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1280 \
    data.truncation=left \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
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
    trainer.experiment_name=qwen3-1.7b-gspo-2a100 \
    wandb_config.project="ADPO-pk-GRPO" \
    wandb_config.name=qwen3-1.7b-gspo-2a100 \
    "$@"  # Pass any additional arguments

echo ""
echo "=========================================="
echo "✅ GSPO Training Complete!"
echo "=========================================="
