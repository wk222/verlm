#!/bin/bash
# Run ALL ADPO variants for comparison
# 依次运行所有变体，方便对比

set -e

echo "=========================================="
echo "ADPO All Variants Batch Runner"
echo "=========================================="
echo ""

# 定义要测试的变体（推荐顺序）
VARIANTS=(
    "pairwise"             # ⭐推荐：DPO风格
    "plackett_luce_approx" # P-L模型近似版
    "direct"               # -q·u + logsumexp
    "infonce"              # 对比学习风格
    "softmax"              # 原始ADPO（baseline）
)

echo "Will run the following variants:"
for v in "${VARIANTS[@]}"; do
    echo "  - $v"
done
echo ""

read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# 运行每个变体
for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "🚀 Running variant: ${VARIANT}"
    echo "=========================================="
    echo ""
    
    bash examples/run_adpo_variants_comparison.sh ${VARIANT}
    
    echo ""
    echo "✅ Completed: ${VARIANT}"
    echo ""
    sleep 5  # Brief pause between runs
done

echo ""
echo "=========================================="
echo "🎉 All variants completed!"
echo "=========================================="
echo ""
echo "Compare results in WandB project: ADPO-Variants-Comparison"

