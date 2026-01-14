#!/bin/bash
set -e  # Exit on error

# Function to clean GPU resources
clean_resources() {
    echo "Cleaning resources..."
    pkill -9 -f "python3 -m verl.trainer.main" || true
    pkill -9 -f "ray" || true
    sleep 10
}

echo "============================================================"
echo "Starting 5-Algorithm Benchmark"
echo "Sequence: AlphaPO(Adaptive) -> AlphaPO(Fixed) -> ADPO-SFT -> GSPO -> ADPO-Softmax"
echo "============================================================"

# 1. Run AlphaPO (Adaptive ESS)
echo "[1/5] Running AlphaPO (Adaptive ESS)..."
clean_resources
bash examples/run_qwen3_alphapo_4gpu.sh
echo "AlphaPO (Adaptive) finished."

# 2. Run AlphaPO (Fixed Alpha=0.6)
echo "[2/5] Running AlphaPO (Fixed Alpha=0.6)..."
clean_resources
bash examples/run_qwen3_alphapo_fixed_4gpu.sh
echo "AlphaPO (Fixed) finished."

# 3. Run ADPO-SFT (No Anchor)
echo "[3/5] Running ADPO-SFT..."
clean_resources
bash examples/run_qwen3_adpo_sft_4gpu.sh
echo "ADPO-SFT finished."

# 4. Run GSPO (Fair Comparison)
echo "[4/5] Running GSPO (Fair)..."
clean_resources
bash examples/run_qwen3_gspo_fair_4gpu.sh
echo "GSPO finished."

# 5. Run ADPO-Softmax (Standard)
echo "[5/5] Running ADPO-Softmax..."
clean_resources
bash examples/run_qwen3_adpo_softmax_4gpu.sh
echo "ADPO-Softmax finished."

echo "============================================================"
echo "All 5 experiments completed successfully!"
echo "============================================================"
