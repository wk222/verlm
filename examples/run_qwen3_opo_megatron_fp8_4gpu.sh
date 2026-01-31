#!/bin/bash
# OPO Training - Qwen3-4B-Thinking on WZX MATH Dataset
# Backend: Megatron-Core with FP8
# Hardware: 4 GPUs (Ada, SM89)

set -e

echo "=========================================="
echo "OPO Megatron FP8 Training - Qwen3-4B on 4 GPUs"
echo "=========================================="

# Check if we're in the verlm directory
if [ ! -d "verl/trainer" ]; then
    echo "❌ Error: Please run this script from the verlm/ directory."
    echo "   Current directory: $(pwd)"
    exit 1
fi

# -----------------------------
# Core environment
# -----------------------------
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 🔴 必须：显式指定 Ada 架构（4070 Ti = sm_89）
# 否则 Megatron unified_memory JIT 会直接 IndexError
export TORCH_CUDA_ARCH_LIST=8.9

# 确保 Ray worker 继承 CUDA 环境
export RAY_DEDUP_LOGS=0
export RAY_worker_register_timeout_seconds=120

# -----------------------------
# Attention / FP8 / TE settings
# -----------------------------
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Transformer Engine FP8 必需
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# 防止 TE / torch.compile 在初始化阶段抢 JIT
export TORCHDYNAMO_DISABLE=1

# -----------------------------
# Configuration
# -----------------------------
CONFIG_NAME="opo_megatron_qwen3_math_fp8"
OUTPUT_DIR="data/Qwen3-4B-OPO-Megatron-FP8"
DATA_DIR="data/math_level3"
N_GPUS=4
MODEL_PATH="Qwen/Qwen3-4B-Thinking-2507-FP8"

echo "Configuration:"
echo "  - Config: ${CONFIG_NAME}"
echo "  - Model: ${MODEL_PATH}"
echo "  - Output: ${OUTPUT_DIR}"
echo "  - Data: ${DATA_DIR}"
echo "  - GPUs: ${N_GPUS}"
echo "  - Algorithm: OPO (Alpha=0.6, Mu=1.0)"
echo "  - Backend: Megatron + FP8"
echo ""

# -----------------------------
# Dataset
# -----------------------------
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "📥 Downloading and preprocessing WZX MATH dataset..."
    python3 examples/data_preprocess/math_wzx_dataset.py \
        --local_save_dir ${DATA_DIR}
    echo ""
else
    echo "✅ WZX MATH dataset already exists at ${DATA_DIR}"
    echo ""
fi

echo "🚀 Starting OPO Megatron FP8 training..."

# -----------------------------
# Launch training
# -----------------------------
python -m verl.trainer.main_adpo \
    --config-name ${CONFIG_NAME} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.project_name="ADPO-GSPO-WZX" \
    trainer.experiment_name=qwen3-4b-opo-megatron-fp8 \
    wandb_config.project="ADPO-GSPO-WZX" \
    wandb_config.name=qwen3-4b-opo-megatron-fp8 \
    "$@"

echo ""
echo "=========================================="
echo "✅ OPO Training Complete!"
echo "=========================================="
