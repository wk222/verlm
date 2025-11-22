# ADPO 实验复现指南

## 复现 Qwen3-1.7B on MATH Dataset

本指南展示如何使用 VERL 复现 Open-R1 ADPO baseline 实验。

### 原始配置来源

基于 `OPENR1_ADPO-VERSION/recipes/Qwen3/adpo/config_qwen3-1_6b.yaml`

### 环境准备

```bash
# 1. 进入 verlm 目录
cd verlm

# 2. 安装依赖
pip install latex2sympy2_extended math_verify

# 3. 验证安装
python examples/test_adpo_installation.py
```

### 快速启动

```bash
# 使用预配置的脚本
bash examples/reproduce_qwen3_math_adpo.sh
```

### 详细配置

#### 配置文件: `verl/trainer/config/adpo_qwen3_math.yaml`

**核心配置对比**:

| 配置项 | 原始 TRL-ADPO | VERL-ADPO |
|--------|---------------|-----------|
| 模型 | Qwen/Qwen3-1.7B | ✓ 相同 |
| 数据集 | MATH-lighteval-level_3 | ✓ 相同 |
| num_generations | 8 | ✓ 相同 |
| tau | 0.8 | ✓ 相同 |
| beta_reward | 0.5 | ✓ 相同 |
| anchor_update_mode | on_policy | ✓ 相同 |
| use_adaptive_tau | True | ✓ 相同 |
| adaptive_tau_alpha | 1.0 | ✓ 相同 |
| adaptive_tau_min | 0.1 | ✓ 相同 |
| learning_rate | 1.5e-5 | ✓ 相同 |
| gradient_accumulation_steps | 16 | ✓ 相同 |
| per_device_train_batch_size | 8 | ✓ 相同 |
| num_train_epochs | 2 | ✓ 相同 |
| vLLM | colocate | ✓ 相同 |
| reward_func | good_accuracy | ✓ 相同 |

### 手动运行

```bash
# 基础命令
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math

# 自定义输出目录
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    trainer.default_local_dir=my_output_dir

# 修改训练参数
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    algorithm.num_generations=16 \
    algorithm.tau=1.0 \
    trainer.per_device_train_batch_size=16
```

### 配置说明

#### 1. 模型配置

```yaml
actor_rollout_ref:
  model:
    path: Qwen/Qwen3-1.7B
  actor:
    model_init_kwargs:
      torch_dtype: bfloat16
      attn_implementation: flash_attention_2
```

#### 2. vLLM 配置

```yaml
rollout:
  use_vllm: true
  vllm_mode: colocate
  vllm_enable_sleep_mode: true
  vllm_gpu_memory_utilization: 0.4
```

#### 3. ADPO 算法配置

```yaml
algorithm:
  adv_estimator: adpo
  num_generations: 8
  tau: 0.8
  anchor_update_mode: on_policy
  use_adaptive_tau: true
  adaptive_tau_alpha: 1.0
  adaptive_tau_min: 0.1
  beta_reward: 0.5
```

#### 4. 奖励函数配置

```yaml
custom_reward_function:
  path: verl/trainer/adpo/reward.py
  name: good_accuracy

reward_model:
  reward_kwargs:
    ngram_size: 4
    max_penalty: -0.5
    penalty_scale_factor: 0.1
```

### 多GPU训练

```bash
# 8 GPU 训练（默认）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash examples/reproduce_qwen3_math_adpo.sh

# 4 GPU 训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    trainer.n_gpus_per_node=4

# 单 GPU 测试
export CUDA_VISIBLE_DEVICES=0
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    trainer.n_gpus_per_node=1 \
    trainer.per_device_train_batch_size=2
```

### 监控训练

#### WandB 配置

```yaml
wandb_config:
  project: open-r1-ADPO
  name: qwen3-1.7b-adpo-baseline
  group: qwen3_adpo_baseline
  tags: [adpo, qwen3, math]
```

查看指标：
```bash
# 登录 WandB
wandb login

# 训练会自动上传到 WandB
```

#### 本地日志

```bash
# 查看训练日志
tail -f data/Qwen3-1.7B-Open-R1-ADPO/logs/train.log

# 查看检查点
ls data/Qwen3-1.7B-Open-R1-ADPO/checkpoint-*
```

### 恢复训练

```bash
# 从检查点恢复
bash examples/reproduce_qwen3_math_adpo.sh \
    trainer.resume_from_checkpoint=data/Qwen3-1.7B-Open-R1-ADPO/checkpoint-1000
```

### 评估模型

```bash
# 在验证集上评估
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    trainer.do_eval=true \
    trainer.do_train=false
```

### 常见问题

#### Q1: 内存不足

**症状**: OOM (Out of Memory) 错误

**解决**:
```bash
# 减小批大小
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    trainer.per_device_train_batch_size=4 \
    algorithm.num_generations=4

# 或增加梯度累积
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    trainer.gradient_accumulation_steps=32
```

#### Q2: vLLM 初始化失败

**症状**: vLLM 相关错误

**解决**:
```bash
# 降低 GPU 利用率
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    rollout.vllm_gpu_memory_utilization=0.3

# 或禁用 sleep mode
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    rollout.vllm_enable_sleep_mode=false
```

#### Q3: 批大小警告

**症状**: Batch size not divisible by num_generations

**解决**: 确保批大小是 `num_generations` 的倍数
```bash
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    trainer.per_device_train_batch_size=16 \
    algorithm.num_generations=8
```

### 实验变体

#### 变体 1: Fixed Anchor

```bash
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    algorithm.anchor_update_mode=fixed \
    algorithm.tau=1.0 \
    trainer.experiment_name=qwen3-adpo-fixed-anchor
```

#### 变体 2: EMA Anchor

```bash
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    algorithm.anchor_update_mode=ema \
    algorithm.ema_alpha=0.99 \
    trainer.experiment_name=qwen3-adpo-ema
```

#### 变体 3: 更多生成数

```bash
python -m verl.trainer.main_adpo \
    --config-name adpo_qwen3_math \
    algorithm.num_generations=16 \
    trainer.per_device_train_batch_size=16 \
    trainer.experiment_name=qwen3-adpo-gen16
```

### 结果对比

预期结果（基于原始 TRL-ADPO 实验）：

| Metric | Epoch 1 | Epoch 2 |
|--------|---------|---------|
| Accuracy | ~0.35 | ~0.45 |
| Mean Reward | ~0.40 | ~0.50 |
| Mean Tau | ~0.6 | ~0.5 |

**注意**: 实际结果可能因随机种子和硬件而略有不同。

### 下一步

1. **查看详细文档**: `verl/trainer/adpo/README.md`
2. **尝试其他配置**: `examples/adpo_example_config.py`
3. **自定义奖励函数**: 参考 `verl/trainer/adpo/reward.py`

### 引用

如果使用此复现脚本，请引用：

```bibtex
@misc{zixian2025adpoanchoreddirectpreference,
    title={ADPO: Anchored Direct Preference Optimization}, 
    author={Wang Zixian},
    year={2025},
    eprint={2510.18913},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2510.18913}, 
}
```

---

**祝实验顺利！** 🚀

