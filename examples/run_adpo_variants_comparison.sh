#!/bin/bash
# ADPO Loss Variants Comparison Script
# 用于对比不同的loss_variant效果
# Usage: ./run_adpo_variants_comparison.sh [variant]
#   variants: softmax, scaled, direct, logit_mse, pairwise, infonce, groupwise_logit

set -e

echo "=========================================="
echo "ADPO Loss Variants Comparison"
echo "=========================================="

# Check directory
if [ ! -d "verl/trainer/adpo" ]; then
    echo "❌ Error: Please run this script from the verlm/ directory."
    exit 1
fi

# Environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
CONFIG_NAME="adpo_qwen3_math_hybrid"
DATA_DIR="data/math_wzx"
N_GPUS=4

# Get variant from argument or use default
VARIANT=${1:-"pairwise"}
OUTPUT_DIR="data/ADPO-${VARIANT}-comparison"

echo ""
echo "🔧 Running ADPO with loss_variant=${VARIANT}"
echo "   Output: ${OUTPUT_DIR}"
echo ""

# Variant-specific settings
case $VARIANT in
    "pairwise")
        # DPO风格 -log σ(u_w - u_l) ⭐推荐
        EXTRA_ARGS="algorithm.loss_variant=pairwise algorithm.tau=0.5 algorithm.beta_reward=0.3"
        ;;
    "plackett_luce")
        # Plackett-Luce模型/ListMLE，完整排序对齐
        EXTRA_ARGS="algorithm.loss_variant=plackett_luce algorithm.tau=0.5 algorithm.beta_reward=0.3"
        ;;
    "plackett_luce_approx")
        # P-L近似版（只看top-k）
        EXTRA_ARGS="algorithm.loss_variant=plackett_luce_approx algorithm.tau=0.5 algorithm.plackett_luce_top_k=3"
        ;;
    "direct")
        # -q·u + logsumexp(u)
        EXTRA_ARGS="algorithm.loss_variant=direct algorithm.tau=0.5 algorithm.beta_reward=0.3"
        ;;
    "infonce")
        # -u_best + logsumexp(u)
        EXTRA_ARGS="algorithm.loss_variant=infonce algorithm.tau=0.5 algorithm.beta_reward=0.3"
        ;;
    "softmax")
        # 原始ADPO（baseline）
        EXTRA_ARGS="algorithm.loss_variant=softmax algorithm.tau=0.3 algorithm.beta_reward=0.25 algorithm.use_adaptive_tau=False"
        ;;
    "scaled")
        # 原始ADPO × 缩放因子
        EXTRA_ARGS="algorithm.loss_variant=scaled algorithm.grad_scale_factor=20.0 algorithm.tau=0.5"
        ;;
    *)
        echo "❌ Unknown variant: ${VARIANT}"
        echo "Available: pairwise, plackett_luce, plackett_luce_approx, direct, infonce, softmax, scaled"
        exit 1
        ;;
esac

# Check data
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "📥 Downloading WZX MATH dataset..."
    python3 examples/data_preprocess/math_wzx_dataset.py --local_save_dir ${DATA_DIR}
fi

# Run training
echo "🚀 Starting training..."
python -m verl.trainer.main_adpo \
    --config-name ${CONFIG_NAME} \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=880 \
    data.max_response_length=1280 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=300 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.project_name="ADPO-Variants-Comparison" \
    trainer.experiment_name=adpo-${VARIANT} \
    wandb_config.project="ADPO-Variants-Comparison" \
    wandb_config.name=adpo-${VARIANT} \
    ${EXTRA_ARGS} \
    "${@:2}"  # Pass additional arguments

echo ""
echo "✅ ADPO (${VARIANT}) Training Complete!"
echo "   Results in: ${OUTPUT_DIR}"

