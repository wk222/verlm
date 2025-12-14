#!/bin/bash
# GSPO Training - Qwen3-1.7B on WZX MATH Dataset
# GSPO = GRPO + 句级概率 (Sentence-level Probability)
# 参考论文: https://arxiv.org/pdf/2507.18071
# Optimized for 4x4090 (24GB VRAM each)

set -e

echo "=========================================="
echo "GSPO Training - Qwen3 on 4x4090"
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

# Configuration
CONFIG_NAME="gspo_qwen3_math_hybrid"
OUTPUT_DIR="data/Qwen3-1.7B-GSPO-WZX"
DATA_DIR="data/math_level3"
N_GPUS=4

echo "Configuration:"
echo "  - Config: ${CONFIG_NAME}"
echo "  - Output: ${OUTPUT_DIR}"
echo "  - Data: ${DATA_DIR}"
echo "  - GPUs: ${CUDA_VISIBLE_DEVICES} (${N_GPUS} GPUs)"
echo "  - Algorithm: GSPO (GRPO + Sentence-level Probability)"
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

echo "🚀 Starting GSPO training..."
echo ""

# ============================================================
# GSPO 全序列级优化版配置说明:
# ============================================================
# 核心优化: 从 Advantage Estimator 到 Policy Loss 全程序列级别计算
#
# 1. adv_estimator: grpo_seq (序列级GRPO，返回 (B,) 而非 (B,T))
#    - 原版 grpo 返回 (B, T) 张量，每个token复制相同advantage值
#    - grpo_seq 返回 (B,) 张量，直接是序列级别
#    - 显存节省: advantages 从 B*T → B (例: 512*1280 → 512)
#
# 2. policy_loss.loss_mode: gspo (自动检测维度，选择最优路径)
#    - 当 advantages 是 (B,) 时：全程序列级计算，高效
#    - 当 advantages 是 (B,T) 时：回退到token级计算，兼容
#    - 与 grpo_seq 配合使用时，性能与 ADPO 相当
#
# 3. 性能提升 (grpo_seq + gspo):
#    - 显存: 与 ADPO 相当 (advantages 从 B*T → B)
#    - 速度: 向量化操作，无Python循环
#    - 吞吐: 可以使用更大的 micro_batch_size
#
# 4. clip_ratio_high: 0.28 (GSPO论文推荐的非对称裁剪)
# ============================================================

python -m verl.trainer.main_ppo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    "$@"

echo ""
echo "=========================================="
echo "✅ GSPO Training Complete!"
echo "=========================================="
