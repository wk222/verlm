#!/bin/bash
# 一键运行OPO/GOPO进行验证
# 顺序：OPO -> GOPO
# 每个实验结束后会自动清理资源

set -e

LOG_DIR="logs_benchmark"
mkdir -p ${LOG_DIR}

echo "============================================================"
echo "🚀 开始运行 OPO/GOPO 算法对比"
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
# 1. OPO
# ------------------------------------------------------------
echo "▶️ [1/2] Running OPO..."
bash examples/run_qwen3_opo_fsdp_full_1.7b_4gpu.sh > ${LOG_DIR}/opo.log 2>&1
echo "✅ OPO 完成! 日志: ${LOG_DIR}/opo.log"
clean_resources

# ------------------------------------------------------------
# 2. GOPO
# ------------------------------------------------------------
echo "▶️ [2/2] Running GOPO..."
bash examples/run_qwen3_gopo_fsdp_full_1.7b_4gpu.sh > ${LOG_DIR}/gopo.log 2>&1
echo "✅ GOPO 完成! 日志: ${LOG_DIR}/gopo.log"
clean_resources


echo "============================================================"
echo "🎉 所有实验运行完成！"
echo "============================================================"
