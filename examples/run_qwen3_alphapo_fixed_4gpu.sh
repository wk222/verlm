#!/bin/bash
# AlphaPO Training (Fixed Alpha) - Qwen3-1.7B on WZX MATH Dataset
# Fixed alpha=0.6 for fair comparison with Adaptive Alpha
# Settings optimized for 4x4090

set -e

echo "=========================================="
echo "AlphaPO Training (Fixed Alpha=0.6) - Qwen3 on 4x4090"
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
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Configuration
CONFIG_NAME="alphapo_fixed_qwen3_math"
OUTPUT_DIR="data/Qwen3-1.7B-AlphaPO-Fixed-WZX"
DATA_DIR="data/math_level3"
N_GPUS=4

echo "Configuration:"
echo "  - Config: ${CONFIG_NAME}"
echo "  - Output: ${OUTPUT_DIR}"
echo "  - Data: ${DATA_DIR}"
echo "  - GPUs: ${CUDA_VISIBLE_DEVICES} (${N_GPUS} GPUs)"
echo "  - Algorithm: AlphaPO (Fixed Alpha=0.6)"
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

echo "🚀 Starting AlphaPO (Fixed) training..."
echo ""

# 使用 main_adpo 运行 AlphaPO Fixed
python -m verl.trainer.main_adpo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    data.train_batch_size=48 \
    data.val_batch_size=24 \
    data.max_prompt_length=1024 \
    data.max_response_length=1280 \
    data.truncation=left \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    algorithm.num_generations=6 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.total_epochs=2 \
    trainer.project_name="ADPO-GSPO-WZX" \
    trainer.experiment_name=qwen3-1.7b-alphapo-fixed-0.6-4gpu \
    wandb_config.project="ADPO-GSPO-WZX" \
    wandb_config.name=qwen3-1.7b-alphapo-fixed-0.6-4gpu \
    "$@"

echo ""
echo "=========================================="
echo "✅ AlphaPO (Fixed) Training Complete!"
echo "=========================================="
