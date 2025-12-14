#!/bin/bash
# ADPO SFT Variant Training - Qwen3 on 4x4090
# SFT Variant: u = log_prob / tau (通过设置 A=0 实现)
#
# Key Features:
# 1. SFT Variant: A=0, 不使用 reference model 的 anchor
# 2. 公式: u = (log_prob - 0*old_log_prob - 0*q - 0) / tau = log_prob / tau
# 3. 与 Standard ADPO (A=1) 对比实验
#
set -e

echo "=========================================="
echo "ADPO SFT Variant - Qwen3 (4x4090)"
echo "=========================================="

if [ ! -d "verl/trainer/adpo" ]; then
    echo "❌ Error: Please run from verlm/ directory."
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
CONFIG_NAME="adpo_qwen3_math_4x4090_sft"
OUTPUT_DIR="data/Qwen3-1.7B-ADPO-SFT-WZX"
DATA_DIR="data/math_level3"
N_GPUS=4

echo "Config: ${CONFIG_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Variant: SFT (A=0, u = log_prob / tau)"

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

echo "✅ SFT Variant Training Complete!"


