# ADPO Integration Summary - VERL 框架

## 概述

本文档总结了在 VERL 框架中添加 ADPO (Anchored Direct Preference Optimization) 训练器的所有更改。

**当前版本使用 on-policy 模式**：`old_log_prob` 作为锚点，无需维护额外的锚点模型，内存效率高。

## 添加的文件

### 1. 核心训练器文件

#### `verl/trainer/adpo/__init__.py`
- ADPO 模块初始化文件
- 导出主要类和函数

#### `verl/trainer/adpo/core_algos.py`
- **核心功能**：
  - `compute_adpo_advantages()`: ADPO 优势估计器
  - `adpo_policy_loss()`: ADPO 策略损失函数
- **关键特性**：
  - 锚点分布计算：`p_θ(i|S) = softmax((s_i - s_anchor_i) / τ)`
  - 自适应温度缩放
  - On-policy 锚点模式（使用 old_log_prob）
  - 可选 KL 惩罚
  - 自动批次截断（当批大小不是 num_generations 的倍数时）

#### `verl/trainer/adpo/ray_trainer.py`
- **RayADPOTrainer** 类：继承自 `RayPPOTrainer`
- **主要功能**：
  - 初始化 ADPO 特定配置
  - 使用 on-policy 锚点模式

#### `verl/trainer/adpo/reward.py`
- **奖励函数模块**
- **主要函数**：
  - `load_reward_manager()`: 加载奖励管理器（包装 PPO 的实现）
  - `good_accuracy()`: 组合准确度奖励函数
    - 使用 VERL 内置的 `prime_math` 进行数学答案验证
    - N-gram 重复惩罚（对错误答案）

#### `verl/trainer/adpo/utils.py`
- **工具函数集合**：
  - `compute_adaptive_tau()`: 自适应温度计算
  - `log_adpo_metrics()`: ADPO 指标记录

### 2. 主入口文件

#### `verl/trainer/main_adpo.py`
- **main()**: Hydra 配置入口
- **run_adpo()**: Ray 集群初始化和训练启动
- **ADPOTaskRunner**: 任务运行器（继承自 `PPOTaskRunner`）
  - 设置分布式训练环境
  - 初始化数据集和奖励函数
  - 创建 RayADPOTrainer 实例
  - 启动训练循环

### 3. 配置文件

#### `verl/trainer/config/adpo_trainer.yaml`
- **完整的 ADPO 配置模板**
- **关键配置项**：
  ```yaml
  algorithm:
    adv_estimator: adpo
    num_generations: 8
    
    # ADPO 核心参数 (on-policy 模式)
    tau: 0.8
    
    # 自适应温度
    use_adaptive_tau: True
    adaptive_tau_alpha: 0.5
    adaptive_tau_min: 0.05
    
    # 奖励计算
    beta_reward: 0.5
    
    # 可选功能
    beta_anchor_kl: 0.0
    drop_all_failed_prompts: False
  ```

### 4. 文档和示例

#### `verl/trainer/adpo/README.md`
- **全面的用户指南**
- **内容包括**：
  - ADPO 核心思想和算法
  - 快速开始指南
  - 完整参数说明
  - 内存优化提示
  - 奖励函数使用指南
  - 与 GRPO 的对比
  - 常见问题解答

#### `examples/run_adpo_gsm8k.sh`
- GSM8K 数据集上的基础 ADPO 训练示例
- 使用 on-policy 锚点模式

#### `examples/adpo_example_config.py`
- **Python 配置示例**
- **提供预定义配置函数**：
  - `get_adpo_base_config()`
  - `get_adpo_on_policy_config()`
  - `get_adpo_with_good_accuracy_config()`
  - `get_adpo_memory_optimized_config()`

## ADPO 核心特性

### 1. On-Policy 锚点模式

当前实现使用 on-policy 模式：
- 使用 `old_log_prob` 作为锚点
- 内存高效（无需额外锚点模型）
- 类似 GRPO 的快速收敛

### 2. 自适应温度缩放

- **基于熵的动态调整**：`τ(x) = max(τ_min, τ_base * (1 - α * H/H_max))`
- **好处**：
  - 简单任务 → 低熵 → 低温度（更聚焦）
  - 困难任务 → 高熵 → 高温度（更探索）

### 3. good_accuracy 奖励函数

```python
# 特性
- 数学答案验证（使用 parse/verify）
- 自动纯数字包装（"0.5" → "{0.5}"）
- N-gram 重复惩罚
- 错误答案的缩放惩罚

# 返回值
- 正确: 1.0
- 错误: 0.0 + penalty_scale * repetition_penalty (≤ 0)
```

## 使用方法

### 快速开始

```bash
# 1. 基础训练（on-policy 模式）
cd verlm
python -m verl.trainer.main_adpo \
    algorithm.adv_estimator=adpo \
    algorithm.tau=0.8 \
    algorithm.num_generations=8

# 2. 使用 good_accuracy 奖励
python -m verl.trainer.main_adpo \
    --config-name adpo_trainer \
    custom_reward_function.path=verl/trainer/adpo/reward.py \
    custom_reward_function.name=good_accuracy \
    reward_model.reward_kwargs.ngram_size=4 \
    reward_model.reward_kwargs.max_penalty=-0.5

# 3. 运行示例脚本
bash examples/run_adpo_gsm8k.sh
```

### 自定义配置

```python
# 使用 Python 配置
from examples.adpo_example_config import get_adpo_with_good_accuracy_config
from verl.trainer.main_adpo import run_adpo

config = get_adpo_with_good_accuracy_config()
run_adpo(config)
```

## 监控指标

ADPO 训练会记录以下特定指标：

| 指标 | 描述 |
|------|------|
| `adpo/anchor_kl` | 当前策略与锚点的 KL 散度 |
| `adpo/mean_tau` | 平均温度值 (如果使用自适应温度) |
| `adpo/loss` | ADPO 损失值 |
| `adpo/kl_penalty` | KL 惩罚值（如果启用） |
| `adpo/dropped_prompts` | 丢弃的 prompt 数量（如果启用） |

## 配置示例对比

### ADPO vs GRPO

```yaml
# GRPO 配置
algorithm:
  adv_estimator: grpo
  epsilon: 0.2  # PPO clipping
  
# ADPO 配置 (on-policy 模式)
algorithm:
  adv_estimator: adpo
  tau: 0.8  # 锚点温度
  use_adaptive_tau: True
```

## 依赖要求

```bash
# 核心依赖（已在 VERL 中）
- torch
- ray
- omegaconf
- hydra-core

# good_accuracy 奖励函数使用 VERL 内置的 prime_math，无需额外依赖
```

## 与现有代码的集成

### 1. 继承 PPO Trainer
- `RayADPOTrainer` 继承 `RayPPOTrainer`
- 复用绝大部分训练逻辑
- 仅覆盖 ADPO 特定行为

### 2. 注册优势估计器和损失函数
```python
# 在 core_algos.py 中
@register_adv_est("adpo")
def compute_adpo_advantages(...)

@register_policy_loss("adpo")
def adpo_policy_loss(...)
```

### 3. 配置继承
- `adpo_trainer.yaml` 基于 `ppo_trainer.yaml`
- 使用相同的 defaults 结构
- 添加 ADPO 特定字段

## 测试建议

### 1. 功能测试
```bash
# 测试 on-policy 模式
python -m verl.trainer.main_adpo algorithm.adv_estimator=adpo

# 测试 good_accuracy 奖励
python -m verl.trainer.main_adpo \
    custom_reward_function.path=verl/trainer/adpo/reward.py \
    custom_reward_function.name=good_accuracy
```

### 2. 批大小测试
```bash
# 确保批大小是 num_generations 的倍数
python -m verl.trainer.main_adpo \
    algorithm.num_generations=8 \
    trainer.per_device_train_batch_size=16  # 16 % 8 == 0
```

### 3. 温度调节测试
```bash
# 测试不同 tau 值
python -m verl.trainer.main_adpo algorithm.tau=0.5
python -m verl.trainer.main_adpo algorithm.tau=1.0
python -m verl.trainer.main_adpo algorithm.tau=2.0
```

## 常见问题

### Q1: tau 参数如何设置？
- **低 tau (0.1-0.5)**：更锐利，强调最优样本
- **中 tau (0.5-1.0)**：平衡，推荐用于大多数任务
- **高 tau (>1.0)**：更平滑，更多探索

### Q2: 何时使用自适应温度？
- 任务难度差异大时
- 希望模型根据样本质量自动调节时
- 建议默认启用（`use_adaptive_tau: True`）

### Q3: 批大小必须是 num_generations 的倍数吗？
- 建议是，但不强制
- ADPO 会自动截断不符合的批次
- 可能会浪费样本，影响效率

## 内存优化

On-policy 模式的优势：
1. **无额外模型存储**：使用 old_log_prob 作为锚点
2. **高效计算**：锚点计算与 PPO 相同
3. **批处理优化**：使用 torch.no_grad() 包裹非梯度计算

推荐配置以减少内存使用：
```yaml
trainer:
  gradient_checkpointing: true
  bf16: true

actor_rollout_ref:
  rollout:
    free_cache_engine: true  # 混合模式
    gpu_memory_utilization: 0.65
```

## 未来改进方向

1. **多任务 ADPO**：不同任务使用不同的参数
2. **Liger Kernel 支持**：实现 `LigerFusedLinearADPOLoss`
3. **更多奖励函数**：添加更多预定义奖励函数
4. **自动超参数调优**：基于任务自动调整 tau 等参数

## 贡献者

- 基于 TRL-ADPO 和 OPENR1_ADPO-VERSION 的实现
- 集成到 VERL 框架

## 引用

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

## 联系方式

如有问题或建议，请：
1. 查看 `verl/trainer/adpo/README.md`
2. 查看示例脚本：`examples/run_adpo_*.sh`
3. 参考配置示例：`examples/adpo_example_config.py`
4. 提交 Issue 或 Pull Request

---

**完成时间**: 2025-11-22  
**状态**: ✅ 所有核心功能已实现并测试

