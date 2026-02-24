#!/bin/bash
set -e

echo "=========================================="
echo "GRPO Training - Qwen3-1.7B FSDP Full FT on 4 GPUs"
echo "=========================================="
echo ""

if [ ! -d "verl/trainer" ]; then
    echo "Error: Please run this script from the verlm/ directory."
    echo "   Current directory: $(pwd)"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Configuration - aligned with OPO/GOPO for fair comparison
CONFIG_NAME="grpo_qwen3_math_hybrid"
OUTPUT_DIR="data/Qwen3-1.7B-GRPO-FSDP-Full-FT"
DATA_DIR="data/math_level3_sampled"
MODEL_PATH="Qwen/Qwen3-1.7B"
N_GPUS=4
SAMPLE_RATIO=0.1
SAMPLE_SEED=42
TOTAL_EPOCHS=8

echo "Configuration:"
echo "  - Config Base: ${CONFIG_NAME}"
echo "  - Output: ${OUTPUT_DIR}"
echo "  - Data: ${DATA_DIR}"
echo "  - GPUs: ${CUDA_VISIBLE_DEVICES} (${N_GPUS} GPUs)"
echo "  - Algorithm: GRPO"
echo ""

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Downloading and preprocessing MATH Level3 dataset (${SAMPLE_RATIO} sample, seed=${SAMPLE_SEED})..."
    python3 examples/data_preprocess/math_level3_dataset.py \
        --local_save_dir ${DATA_DIR} \
        --sample_ratio ${SAMPLE_RATIO} \
        --seed ${SAMPLE_SEED}
    echo ""
else
    echo "MATH Level3 sampled dataset already exists at ${DATA_DIR}"
    echo ""
fi

VAL_DATA_DIR="data/math_level4_sampled"
if [ ! -f "${VAL_DATA_DIR}/test.parquet" ]; then
    echo "Downloading and preprocessing MATH Level4 dataset for validation..."
    python3 examples/data_preprocess/math_level4_dataset.py \
        --local_save_dir ${VAL_DATA_DIR} \
        --val_sample_size 100 --seed 42
    echo ""
else
    echo "MATH Level4 validation dataset already exists at ${VAL_DATA_DIR}"
    echo ""
fi

echo "Starting GRPO training..."
echo ""

python3 -m verl.trainer.main_ppo \
    --config-name ${CONFIG_NAME} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.project_name=GOPO-Benchmark \
    trainer.experiment_name=grpo \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.save_each_epoch=false \
    trainer.push_to_hub=false \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${VAL_DATA_DIR}/test.parquet \
    data.train_batch_size=48 \
    data.val_batch_size=24 \
    data.max_prompt_length=1024 \
    data.max_response_length=1280 \
    data.truncation=left \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.load_format=safetensors \
    +actor_rollout_ref.rollout.layered_summon=True \
    +algorithm.num_generations=6 \
    wandb_config.project=GOPO-Benchmark \
    wandb_config.name=grpo \
    wandb_config.group=qwen3_benchmark \
    "$@"
