#!/bin/bash
# ADPO Quick Start Script
# This script provides a simple way to test ADPO with minimal configuration

set -e  # Exit on error

echo "=========================================="
echo "ADPO Quick Start Script"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found. Please install Python 3.8+."
    exit 1
fi

# Check if we're in the verlm directory
if [ ! -d "verl/trainer/adpo" ]; then
    echo "❌ Error: Please run this script from the verlm/ directory."
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Display menu
echo "Choose an ADPO training mode:"
echo "1) On-Policy (like GRPO) - Recommended for beginners"
echo "2) Fixed Anchor (Standard ADPO)"
echo "3) EMA Anchor (Dynamic updates)"
echo "4) KL-Triggered Anchor (Adaptive)"
echo "5) With good_accuracy reward"
echo ""
read -p "Enter your choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting ADPO with On-Policy anchor..."
        python -m verl.trainer.main_adpo \
            --config-name adpo_trainer \
            algorithm.adv_estimator=adpo \
            algorithm.anchor_update_mode=on_policy \
            algorithm.num_generations=4 \
            algorithm.tau=0.8 \
            trainer.total_epochs=5 \
            trainer.experiment_name=adpo_quickstart_on_policy
        ;;
    2)
        echo ""
        echo "🚀 Starting ADPO with Fixed anchor..."
        python -m verl.trainer.main_adpo \
            --config-name adpo_trainer \
            algorithm.adv_estimator=adpo \
            algorithm.anchor_update_mode=fixed \
            algorithm.num_generations=4 \
            algorithm.tau=1.0 \
            trainer.total_epochs=5 \
            trainer.experiment_name=adpo_quickstart_fixed
        ;;
    3)
        echo ""
        echo "🚀 Starting ADPO with EMA anchor..."
        python -m verl.trainer.main_adpo \
            --config-name adpo_trainer \
            algorithm.adv_estimator=adpo \
            algorithm.anchor_update_mode=ema \
            algorithm.ema_alpha=0.99 \
            algorithm.num_generations=4 \
            algorithm.tau=0.8 \
            trainer.total_epochs=5 \
            trainer.experiment_name=adpo_quickstart_ema
        ;;
    4)
        echo ""
        echo "🚀 Starting ADPO with KL-triggered anchor..."
        python -m verl.trainer.main_adpo \
            --config-name adpo_trainer \
            algorithm.adv_estimator=adpo \
            algorithm.anchor_update_mode=kl_triggered \
            algorithm.kl_threshold=0.1 \
            algorithm.num_generations=4 \
            algorithm.tau=0.8 \
            trainer.total_epochs=5 \
            trainer.experiment_name=adpo_quickstart_kl_triggered
        ;;
    5)
        echo ""
        echo "🚀 Starting ADPO with good_accuracy reward..."
        echo "Note: Uses VERL's built-in sympy-based math verification"
        python -m verl.trainer.main_adpo \
            --config-name adpo_trainer \
            algorithm.adv_estimator=adpo \
            algorithm.anchor_update_mode=on_policy \
            algorithm.num_generations=4 \
            algorithm.tau=0.8 \
            algorithm.drop_all_failed_prompts=True \
            custom_reward_function.path=verl/trainer/adpo/reward.py \
            custom_reward_function.name=good_accuracy \
            reward_model.reward_kwargs.ngram_size=4 \
            reward_model.reward_kwargs.max_penalty=-0.5 \
            reward_model.reward_kwargs.penalty_scale_factor=0.1 \
            trainer.total_epochs=5 \
            trainer.experiment_name=adpo_quickstart_good_accuracy
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again and choose 1-5."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ ADPO Training Complete!"
echo "=========================================="
echo ""
echo "📊 To view training metrics:"
echo "   - Check logs in the output directory"
echo "   - View WandB dashboard (if enabled)"
echo ""
echo "📖 For more information:"
echo "   - Read: verl/trainer/adpo/README.md"
echo "   - Examples: examples/run_adpo_*.sh"
echo "   - Config: examples/adpo_example_config.py"

