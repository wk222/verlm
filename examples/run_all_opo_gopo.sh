#!/bin/bash
# GOPO Benchmark: 一键运行 5 种算法对比
# 顺序: GRPO -> GSPO -> DAPO -> OPO -> GOPO
# GOPO 放最后以便观察最终结果
# 每个实验结束后自动清理资源

set -eo pipefail

LOG_DIR="logs_benchmark"
mkdir -p ${LOG_DIR}

echo "============================================================"
echo "  GOPO Benchmark - 5 Algorithm Comparison"
echo "============================================================"
echo "  Algorithms: GRPO, GSPO, DAPO, OPO, GOPO"
echo "  Project:    GOPO-Benchmark (wandb)"
echo "  Logs:       ${LOG_DIR}/"
echo "============================================================"
echo ""

clean_resources() {
    echo "Syncing wandb..."
    wandb sync --sync-all > /dev/null 2>&1 || true
    echo "Cleaning resources..."
    ray stop --force || true
    pkill -f "verl.trainer" || true
    pkill -f "vllm" || true
    sleep 10
    echo "Resources cleaned."
    echo ""
}

# Initial cleanup
clean_resources

# ------------------------------------------------------------
# 1. GRPO
# ------------------------------------------------------------
echo "[1/5] Running GRPO..."
bash examples/run_qwen3_grpo_fsdp_full_1.7b_4gpu.sh 2>&1 | tee ${LOG_DIR}/grpo.log
echo "GRPO done! Log: ${LOG_DIR}/grpo.log"
clean_resources

# ------------------------------------------------------------
# 2. GSPO
# ------------------------------------------------------------
echo "[2/5] Running GSPO..."
bash examples/run_qwen3_gspo_fsdp_full_1.7b_4gpu.sh 2>&1 | tee ${LOG_DIR}/gspo.log
echo "GSPO done! Log: ${LOG_DIR}/gspo.log"
clean_resources

# ------------------------------------------------------------
# 3. DAPO
# ------------------------------------------------------------
echo "[3/5] Running DAPO..."
bash examples/run_qwen3_dapo_fsdp_full_1.7b_4gpu.sh 2>&1 | tee ${LOG_DIR}/dapo.log
echo "DAPO done! Log: ${LOG_DIR}/dapo.log"
clean_resources

# ------------------------------------------------------------
# 4. OPO
# ------------------------------------------------------------
echo "[4/5] Running OPO..."
bash examples/run_qwen3_opo_fsdp_full_1.7b_4gpu.sh 2>&1 | tee ${LOG_DIR}/opo.log
echo "OPO done! Log: ${LOG_DIR}/opo.log"
clean_resources

# ------------------------------------------------------------
# 5. GOPO (Main Algorithm)
# ------------------------------------------------------------
echo "[5/5] Running GOPO..."
bash examples/run_qwen3_gopo_fsdp_full_1.7b_4gpu.sh 2>&1 | tee ${LOG_DIR}/gopo.log
echo "GOPO done! Log: ${LOG_DIR}/gopo.log"
clean_resources

echo "============================================================"
echo "  All 5 experiments completed!"
echo "  Check wandb project: GOPO-Benchmark"
echo "============================================================"
