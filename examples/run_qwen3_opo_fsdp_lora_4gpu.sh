#!/bin/bash
set -e

echo "=========================================="
echo "OPO Training - Qwen3-4B FSDP+LoRA on 4 GPUs"
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
# Fix for Megatron JIT compile error
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Configuration
CONFIG_NAME="opo_qwen3_math.yaml"
OUTPUT_DIR="data/Qwen3-4B-OPO-FSDP-LoRA"
DATA_DIR="data/math_level3"
# Attempt to download standard model as requested
MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507"
N_GPUS=4

echo "Configuration:"
echo "  - Config Base: ${CONFIG_NAME}"
echo "  - Output: ${OUTPUT_DIR}"
echo "  - Data: ${DATA_DIR}"
echo "  - GPUs: ${CUDA_VISIBLE_DEVICES} (${N_GPUS} GPUs)"
echo "  - Backend: FSDP + LoRA (BF16)"
echo "  - Model: Qwen3-4B (Download & Train)"
echo ""

# Download and preprocess MATH Level3 dataset if not exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "📥 Downloading and preprocessing MATH Level3 dataset..."
    python3 examples/data_preprocess/math_level3_dataset.py \
        --local_save_dir ${DATA_DIR}
    echo ""
else
    echo "✅ MATH Level3 dataset already exists at ${DATA_DIR}"
    echo ""
fi

echo "🚀 Starting OPO training..."
echo ""

python3 -m verl.trainer.main_adpo \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.project_name=ADPO-GSPO-WZX \
    trainer.experiment_name=qwen3-4b-opo-fsdp-lora \
    trainer.n_gpus_per_node=${N_GPUS} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.actor.policy_loss.num_generations=12 \
    actor_rollout_ref.rollout.n=12 \
    algorithm.num_generations=12 \
    actor_rollout_ref.rollout.load_format=safetensors \
    +actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    --config-name ${CONFIG_NAME} \
    "$@"
