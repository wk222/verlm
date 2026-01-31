#!/bin/bash
# ============================================================
# Fair Comparison: OPO vs GSPO vs GRPO vs DAPO
# ============================================================
# 所有算法使用相同的基础配置，仅在算法核心差异上有所不同
# 基准配置来源: opo_qwen3_math.yaml
#
# 统一参数:
#   - Model: Qwen/Qwen3-1.7B
#   - Dataset: math_level3
#   - Epochs: 2
#   - Batch Size: 48 (train), 24 (val)
#   - Mini Batch: 24
#   - Micro Batch per GPU: 6
#   - Learning Rate: 2e-6
#   - Rollout N: 6
#   - Max Prompt Length: 1024
#   - Max Response Length: 1280
#   - GPU Memory Utilization: 0.55
#   - Gradient Checkpointing: True
#   - FSDP2 Strategy
# ============================================================

set -e
#sudo mount -o remount,size=64G /dev/shm
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the verlm directory
if [ ! -d "verl/trainer" ]; then
    print_error "Please run this script from the verlm/ directory."
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# ============================================================
# Unified Configuration (Based on OPO baseline)
# ============================================================
MODEL_PATH="Qwen/Qwen3-1.7B"
DATA_DIR="data/math_level3"
N_GPUS=4

# Training hyperparameters (unified)
EPOCHS=8
TRAIN_BATCH_SIZE=48
VAL_BATCH_SIZE=24
MINI_BATCH_SIZE=24
MICRO_BATCH_SIZE_PER_GPU=6
LEARNING_RATE="2e-6"
ROLLOUT_N=6
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=1280
GPU_MEMORY_UTIL=0.55
GLOBAL_BATCH_SIZE=32

# WandB project
WANDB_PROJECT="Fair-Comparison-OPO-GSPO-GRPO-DAPO"

# ============================================================
# Algorithm Selection
# ============================================================
usage() {
    echo "Usage: $0 <algorithm> [options]"
    echo ""
    echo "Algorithms:"
    echo "  opo   - Orthogonalized Policy Optimization (ADPO trainer)"
    echo "  gspo  - GRPO + Sentence-level Probability (PPO trainer)"
    echo "  grpo  - Group Relative Policy Optimization (PPO trainer)"
    echo "  dapo  - Dynamic Advantage Preference Optimization (PPO trainer)"
    echo "  all   - Run all algorithms sequentially"
    echo ""
    echo "Options:"
    echo "  Additional Hydra overrides can be passed after the algorithm name"
    echo ""
    echo "Examples:"
    echo "  $0 opo"
    echo "  $0 gspo"
    echo "  $0 all"
    echo "  $0 grpo trainer.total_epochs=4"
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

ALGORITHM=$1
shift  # Remove the algorithm from arguments

# ============================================================
# Data Preparation
# ============================================================
prepare_data() {
    if [ ! -f "${DATA_DIR}/train.parquet" ]; then
        print_header "Downloading and preprocessing MATH Level3 dataset"
        python3 examples/data_preprocess/math_level3_dataset.py \
            --local_save_dir ${DATA_DIR}
    else
        print_success "MATH Level3 dataset already exists at ${DATA_DIR}"
    fi
}

# ============================================================
# Common Overrides (Applied to all algorithms)
# ============================================================
COMMON_OVERRIDES=(
    "data.train_files=${DATA_DIR}/train.parquet"
    "data.val_files=${DATA_DIR}/train.parquet"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.truncation=left"
    "data.shuffle=True"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.use_shm=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL}"
    "actor_rollout_ref.rollout.enforce_eager=False"
    "actor_rollout_ref.rollout.enable_chunked_prefill=True"
    "actor_rollout_ref.rollout.enable_prefix_caching=True"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32"
    "actor_rollout_ref.rollout.max_num_seqs=256"
    #"actor_rollout_ref.rollout.load_format=safetensors"  # vLLM auto-detects
    #"+actor_rollout_ref.rollout.layered_summon=True"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.actor.strategy=fsdp2"
    "actor_rollout_ref.actor.use_dynamic_bsz=False"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.use_orig_params=True"
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
    "trainer.total_epochs=${EPOCHS}"
    "trainer.n_gpus_per_node=${N_GPUS}"
    "trainer.test_freq=-1"
    "trainer.val_before_train=False"
    "trainer.log_val_generations=0"
    "trainer.save_freq=-1"
    "trainer.save_each_epoch=False"
    "trainer.push_to_hub=False"
    "trainer.hub_final_upload=False"
    "algorithm.global_batch_size=${GLOBAL_BATCH_SIZE}"
    "wandb_config.project=${WANDB_PROJECT}"
)

# ============================================================
# OPO Training (uses main_adpo)
# ============================================================
run_opo() {
    local OUTPUT_DIR="data/fair_comparison/Qwen3-1.7B-OPO"
    local EXP_NAME="qwen3-1.7b-opo-fair"
    
    print_header "OPO Training - Orthogonalized Policy Optimization"
    echo "Output: ${OUTPUT_DIR}"
    echo "Experiment: ${EXP_NAME}"
    
    python3 -m verl.trainer.main_adpo \
        --config-name opo_qwen3_math.yaml \
        "${COMMON_OVERRIDES[@]}" \
        "trainer.default_local_dir=${OUTPUT_DIR}" \
        "trainer.project_name=${WANDB_PROJECT}" \
        "trainer.experiment_name=${EXP_NAME}" \
        "actor_rollout_ref.actor.policy_loss.num_generations=${ROLLOUT_N}" \
        "algorithm.num_generations=${ROLLOUT_N}" \
        "wandb_config.name=${EXP_NAME}" \
        "wandb_config.group=fair_comparison" \
        "$@"
    
    print_success "OPO Training Complete!"
}

# ============================================================
# GSPO Training (uses main_ppo)
# ============================================================
run_gspo() {
    local OUTPUT_DIR="data/fair_comparison/Qwen3-1.7B-GSPO"
    local EXP_NAME="qwen3-1.7b-gspo-fair"
    
    print_header "GSPO Training - GRPO + Sentence-level Probability"
    echo "Output: ${OUTPUT_DIR}"
    echo "Experiment: ${EXP_NAME}"
    
    python3 -m verl.trainer.main_ppo \
        --config-name gspo_qwen3_math_hybrid \
        "${COMMON_OVERRIDES[@]}" \
        "+trainer.default_local_dir=${OUTPUT_DIR}" \
        "trainer.project_name=${WANDB_PROJECT}" \
        "trainer.experiment_name=${EXP_NAME}" \
        "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" \
        "actor_rollout_ref.actor.loss_agg_mode=token-mean" \
        "algorithm.adv_estimator=grpo_seq" \
        "wandb_config.name=${EXP_NAME}" \
        "wandb_config.group=fair_comparison" \
        "$@"
    
    print_success "GSPO Training Complete!"
}

# ============================================================
# GRPO Training (uses main_ppo)
# ============================================================
run_grpo() {
    local OUTPUT_DIR="data/fair_comparison/Qwen3-1.7B-GRPO"
    local EXP_NAME="qwen3-1.7b-grpo-fair"
    
    print_header "GRPO Training - Group Relative Policy Optimization"
    echo "Output: ${OUTPUT_DIR}"
    echo "Experiment: ${EXP_NAME}"
    
    python3 -m verl.trainer.main_ppo \
        --config-name grpo_qwen3_math_hybrid \
        "${COMMON_OVERRIDES[@]}" \
        "trainer.default_local_dir=${OUTPUT_DIR}" \
        "trainer.project_name=${WANDB_PROJECT}" \
        "trainer.experiment_name=${EXP_NAME}" \
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla" \
        "actor_rollout_ref.actor.loss_agg_mode=token-mean" \
        "actor_rollout_ref.actor.clip_ratio=0.2" \
        "actor_rollout_ref.actor.clip_ratio_low=0.2" \
        "actor_rollout_ref.actor.clip_ratio_high=0.2" \
        "algorithm.adv_estimator=grpo" \
        "algorithm.norm_adv_by_std_in_grpo=True" \
        "wandb_config.name=${EXP_NAME}" \
        "wandb_config.group=fair_comparison" \
        "$@"
    
    print_success "GRPO Training Complete!"
}

# ============================================================
# DAPO Training (uses main_ppo)
# ============================================================
run_dapo() {
    local OUTPUT_DIR="data/fair_comparison/Qwen3-1.7B-DAPO"
    local EXP_NAME="qwen3-1.7b-dapo-fair"
    
    print_header "DAPO Training - Dynamic Advantage Preference Optimization"
    echo "Output: ${OUTPUT_DIR}"
    echo "Experiment: ${EXP_NAME}"
    
    python3 -m verl.trainer.main_ppo \
        --config-name dapo_qwen3_math_hybrid \
        "${COMMON_OVERRIDES[@]}" \
        "trainer.default_local_dir=${OUTPUT_DIR}" \
        "trainer.project_name=${WANDB_PROJECT}" \
        "trainer.experiment_name=${EXP_NAME}" \
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla" \
        "actor_rollout_ref.actor.loss_agg_mode=token-mean" \
        "actor_rollout_ref.actor.clip_ratio=0.2" \
        "actor_rollout_ref.actor.clip_ratio_low=0.0" \
        "actor_rollout_ref.actor.clip_ratio_high=0.28" \
        "algorithm.adv_estimator=grpo" \
        "algorithm.norm_adv_by_std_in_grpo=False" \
        "wandb_config.name=${EXP_NAME}" \
        "wandb_config.group=fair_comparison" \
        "$@"
    
    print_success "DAPO Training Complete!"
}

# ============================================================
# Run All Algorithms
# ============================================================
run_all() {
    print_header "Running All Algorithms for Fair Comparison"
    
    echo "Will run in order: OPO -> GSPO -> GRPO -> DAPO"
    echo ""
    
    run_opo "$@"
    run_gspo "$@"
    run_grpo "$@"
    run_dapo "$@"
    
    print_header "All Training Complete!"
    echo "Results saved to: data/fair_comparison/"
    echo ""
    echo "Experiments logged to WandB project: ${WANDB_PROJECT}"
}

# ============================================================
# Main Execution
# ============================================================
print_header "Fair Comparison Experiment Suite"
echo "Configuration Summary:"
echo "  - Model:          ${MODEL_PATH}"
echo "  - Dataset:        ${DATA_DIR}"
echo "  - Epochs:         ${EPOCHS}"
echo "  - Batch Size:     ${TRAIN_BATCH_SIZE} (train) / ${VAL_BATCH_SIZE} (val)"
echo "  - Mini Batch:     ${MINI_BATCH_SIZE}"
echo "  - Learning Rate:  ${LEARNING_RATE}"
echo "  - Rollout N:      ${ROLLOUT_N}"
echo "  - GPUs:           ${N_GPUS}"
echo ""

# Prepare data first
prepare_data

# Run the selected algorithm(s)
# Support multiple algorithms: grpo,dapo or grpo dapo
run_algorithm() {
    local algo=$1
    shift
    case $algo in
        opo)
            run_opo "$@"
            ;;
        gspo)
            run_gspo "$@"
            ;;
        grpo)
            run_grpo "$@"
            ;;
        dapo)
            run_dapo "$@"
            ;;
        all)
            run_all "$@"
            ;;
        *)
            print_error "Unknown algorithm: ${algo}"
            usage
            exit 1
            ;;
    esac
}

# Parse comma-separated algorithms (e.g., grpo,dapo)
IFS=',' read -ra ALGO_ARRAY <<< "$ALGORITHM"

for algo in "${ALGO_ARRAY[@]}"; do
    run_algorithm "$algo" "$@"
done
