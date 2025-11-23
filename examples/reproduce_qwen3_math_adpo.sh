#!/bin/bash
# Reproduce Open-R1 ADPO Baseline - Qwen3-1.7B on MATH Dataset
# 
# This script reproduces the ADPO training configuration from:
# https://github.com/your-repo/OPENR1_ADPO-VERSION
#
# Dataset: watermelonhjg/MATH-lighteval-level_3
# Model: Qwen/Qwen3-1.7B
# Method: ADPO (Anchored Direct Preference Optimization)

set -e  # Exit on error

echo "=========================================="
echo "ADPO Reproduction - Qwen3 on MATH"
echo "=========================================="
echo ""

# Check if we're in the verlm directory
if [ ! -d "verl/trainer/adpo" ]; then
    echo "❌ Error: Please run this script from the verlm/ directory."
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 4 GPUs

# Configuration
CONFIG_NAME="adpo_qwen3_math"
OUTPUT_DIR="data/Qwen3-1.7B-Open-R1-ADPO"
N_GPUS=4

echo "Configuration:"
echo "  - Config: ${CONFIG_NAME}"
echo "  - Output: ${OUTPUT_DIR}"
echo "  - GPUs: ${CUDA_VISIBLE_DEVICES} (${N_GPUS} GPUs)"
echo ""

# Dependencies are handled by VERL's prime_math (sympy-based)
echo "✅ Using VERL's built-in sympy-based math verification"
echo ""

# Run ADPO training
echo "🚀 Starting ADPO training..."
echo ""

python -m verl.trainer.main_adpo \
    --config-name ${CONFIG_NAME} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.experiment_name=qwen3-1.7b-adpo-math-reproduction \
    wandb_config.name=qwen3-1.7b-adpo-math-reproduction \
    "$@"  # Pass any additional arguments

echo ""
echo "=========================================="
echo "✅ ADPO Training Complete!"
echo "=========================================="
echo ""
echo "📊 Results saved to: ${OUTPUT_DIR}"
echo ""
echo "📈 To view metrics:"
echo "   - WandB: Check your wandb dashboard"
echo "   - Logs: ${OUTPUT_DIR}/logs"
echo ""
echo "🔍 To resume training:"
echo "   bash examples/reproduce_qwen3_math_adpo.sh \\"
echo "       trainer.resume_from_checkpoint=${OUTPUT_DIR}/checkpoint-XXX"

