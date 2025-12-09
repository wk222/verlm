#!/bin/bash
# Run ADPO with Plackett-Luce loss, 16 generations, on 4x RTX 4090 (24GB)

set -e

if [ ! -d "verl/trainer/adpo" ]; then
  echo "❌ Please run from project root (verlm/)"; pwd; exit 1; fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG_NAME="adpo_qwen3_math_4x4090_pl16"
OUTPUT_DIR="data/Qwen3-1.7B-ADPO-P-L-16gen-4x4090"
DATA_DIR="data/math_wzx"
N_GPUS=4

echo "Config: ${CONFIG_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Data: ${DATA_DIR}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"

# Download dataset if missing
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
  echo "📥 Preparing WZX MATH dataset..."
  python3 examples/data_preprocess/math_wzx_dataset.py --local_save_dir ${DATA_DIR}
fi

# Notes for stability on 24GB cards:
# - train_batch_size=32, num_generations=16 -> 512 sequences per batch (exactly max_num_seqs=128 * 4 GPUs)
# - ppo_micro_batch_size_per_gpu=4 keeps memory in check
# - log_prob_micro_batch_size_per_gpu=8 is conservative

python -m verl.trainer.main_adpo \
  --config-name ${CONFIG_NAME} \
  data.train_files=${DATA_DIR}/train.parquet \
  data.val_files=${DATA_DIR}/train.parquet \
  data.train_batch_size=32 \
  data.val_batch_size=32 \
  data.max_prompt_length=1024 \
  data.max_response_length=1280 \
  data.truncation=left \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.default_local_dir=${OUTPUT_DIR} \
  trainer.project_name="ADPO-P-L" \
  trainer.experiment_name=qwen3-1.7b-adpo-pl16-4x4090 \
  wandb_config.project="ADPO-P-L" \
  wandb_config.name=qwen3-1.7b-adpo-pl16-4x4090 \
  "$@"

echo "✅ Done"
