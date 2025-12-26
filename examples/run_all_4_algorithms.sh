#!/bin/bash
# 一键运行四个算法的公平对比脚本
# 顺序：AlphaPO -> ADPO-SFT -> GSPO -> ADPO-Softmax
# 每个实验结束后会自动清理资源

set -e

LOG_DIR="logs_benchmark"
mkdir -p ${LOG_DIR}

echo "============================================================"
echo "🚀 开始运行 ADPO/GSPO 四大算法公平对比 Benchmark"
echo "============================================================"
echo "日志将保存在: ${LOG_DIR}"
echo ""

clean_resources() {
    echo "🧹 清理资源..."
    ray stop --force || true
    # 尝试杀掉残留的 python 进程 (小心误杀，这里只针对 verl 相关的)
    pkill -f "verl.trainer" || true
    pkill -f "vllm" || true
    sleep 10
    echo "✅ 资源清理完成"
    echo ""
}

# 初始清理
clean_resources

# ------------------------------------------------------------
# 1. AlphaPO
# ------------------------------------------------------------
echo "▶️ [1/4] Running AlphaPO..."
bash examples/run_qwen3_alphapo_4gpu.sh > ${LOG_DIR}/alphapo.log 2>&1
echo "✅ AlphaPO 完成! 日志: ${LOG_DIR}/alphapo.log"
clean_resources

# ------------------------------------------------------------
# 2. ADPO-SFT
# ------------------------------------------------------------
echo "▶️ [2/4] Running ADPO-SFT (Unanchored)..."
bash examples/run_qwen3_adpo_sft_4gpu.sh > ${LOG_DIR}/adpo_sft.log 2>&1
echo "✅ ADPO-SFT 完成! 日志: ${LOG_DIR}/adpo_sft.log"
clean_resources

# ------------------------------------------------------------
# 3. GSPO (Baseline)
# ------------------------------------------------------------
echo "▶️ [3/4] Running GSPO (Fair Baseline)..."
bash examples/run_qwen3_gspo_fair_4gpu.sh > ${LOG_DIR}/gspo.log 2>&1
echo "✅ GSPO 完成! 日志: ${LOG_DIR}/gspo.log"
clean_resources

# ------------------------------------------------------------
# 4. ADPO-Softmax (Standard)
# ------------------------------------------------------------
echo "▶️ [4/4] Running ADPO-Softmax (Standard)..."
bash examples/run_qwen3_adpo_softmax_4gpu.sh > ${LOG_DIR}/adpo_softmax.log 2>&1
echo "✅ ADPO-Softmax 完成! 日志: ${LOG_DIR}/adpo_softmax.log"
clean_resources

echo "============================================================"
echo "🎉 所有实验运行完成！"
echo "============================================================"

