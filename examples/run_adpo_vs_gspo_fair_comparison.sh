#!/bin/bash
# ============================================================
# ADPO vs GSPO 公平对比实验
# ============================================================
# 统一配置：
#   - 数据集: math_level3 (2000+ 条)
#   - train_batch_size: 48
#   - ppo_mini_batch_size: 24
#   - num_generations: 6
#   - max_prompt_length: 1024
#   - max_response_length: 1280
#   - lr: 1e-6
#   - epochs: 2
# ============================================================

set -e

echo "============================================================"
echo "ADPO vs GSPO 公平对比实验"
echo "============================================================"
echo ""

# Check directory
if [ ! -d "verl/trainer" ]; then
    echo "❌ Error: Please run from verlm/ directory."
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 统一配置
DATA_DIR="data/math_level3"
N_GPUS=4
TRAIN_BATCH_SIZE=48
VAL_BATCH_SIZE=24
PPO_MINI_BATCH_SIZE=24
PPO_MICRO_BATCH_SIZE_PER_GPU=6
NUM_GENERATIONS=6
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=1280
LR=1e-6

echo "统一配置:"
echo "  - 数据集: ${DATA_DIR}"
echo "  - train_batch_size: ${TRAIN_BATCH_SIZE}"
echo "  - ppo_mini_batch_size: ${PPO_MINI_BATCH_SIZE}"
echo "  - num_generations: ${NUM_GENERATIONS}"
echo "  - max_prompt_length: ${MAX_PROMPT_LENGTH}"
echo "  - max_response_length: ${MAX_RESPONSE_LENGTH}"
echo "  - lr: ${LR}"
echo ""

# Ensure dataset exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "📥 Preparing math_level3 dataset..."
    python3 examples/data_preprocess/math_level3_dataset.py --local_save_dir ${DATA_DIR}
fi

# ============================================================
# 实验 1: ADPO Softmax
# ============================================================
echo ""
echo "============================================================"
echo "🚀 实验 1/2: ADPO Softmax"
echo "============================================================"
echo ""

ADPO_OUTPUT_DIR="data/fair_comparison/ADPO-Softmax"

python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math_4x4090_softmax \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.n=${NUM_GENERATIONS} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir=${ADPO_OUTPUT_DIR} \
    trainer.project_name="ADPO-vs-GSPO" \
    trainer.experiment_name=adpo-softmax-fair \
    wandb_config.project="ADPO-vs-GSPO" \
    wandb_config.name=adpo-softmax-fair \
    wandb_config.group=fair_comparison \
    "$@"

echo ""
echo "✅ ADPO Softmax 完成!"
echo ""

# ============================================================
# 实验 2: GSPO
# ============================================================
echo ""
echo "============================================================"
echo "🚀 实验 2/2: GSPO"
echo "============================================================"
echo ""

GSPO_OUTPUT_DIR="data/fair_comparison/GSPO"

python -m verl.trainer.main_ppo \
    --config-name gspo_qwen3_math_hybrid \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/train.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.n=${NUM_GENERATIONS} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir=${GSPO_OUTPUT_DIR} \
    trainer.project_name="ADPO-vs-GSPO" \
    trainer.experiment_name=gspo-fair \
    wandb_config.project="ADPO-vs-GSPO" \
    wandb_config.name=gspo-fair \
    wandb_config.group=fair_comparison \
    "$@"

echo ""
echo "✅ GSPO 完成!"
echo ""

echo "============================================================"
echo "🎉 公平对比实验全部完成!"
echo "============================================================"
echo ""
echo "结果目录:"
echo "  - ADPO Softmax: ${ADPO_OUTPUT_DIR}"
echo "  - GSPO: ${GSPO_OUTPUT_DIR}"
echo ""
echo "WandB 项目: ADPO-vs-GSPO (group: fair_comparison)"
echo ""

