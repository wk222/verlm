#!/bin/bash
# ADPO Decoupled Loss Training - Qwen3 on 4x4090
# Comparison Run: Decoupled Loss vs Softmax
#
# Key Features:
# 1. Decoupled Dataflow: Uses precomputed Q to break group constraints
# 2. Optimized Batching: ppo_mini_batch_size=60 (not strictly divisible by n=6 per GPU, but handled by decoupled logic)

set -e

echo "=========================================="
echo "ADPO Decoupled Training - Qwen3 (4x4090)"
echo "=========================================="

if [ ! -d "verl/trainer/adpo" ]; then
    echo "❌ Error: Please run from verlm/ directory."
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
CONFIG_NAME="adpo_qwen3_math_4x4090_decoupled"
OUTPUT_DIR="data/Qwen3-1.7B-ADPO-Decoupled-WZX"
DATA_DIR="data/math_level3"
N_GPUS=4

echo "Config: ${CONFIG_NAME}"
echo "Output: ${OUTPUT_DIR}"

# Ensure dataset exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    python3 examples/data_preprocess/math_level3_dataset.py --local_save_dir ${DATA_DIR}
fi

# Run Training
# Note: overrides here are minimal as most settings are in the YAML
python -m verl.trainer.main_adpo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    trainer.n_gpus_per_node=${N_GPUS} \
    +trainer.default_local_dir=${OUTPUT_DIR} \
    "$@"

echo "✅ Decoupled Training Complete!"

