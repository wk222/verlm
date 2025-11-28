# ADPO (Anchored Direct Preference Optimization) Trainer

ADPO 是一种新的强化学习算法，通过锚点分布提供几何稳定性来优化策略。

## 核心思想

ADPO 使用锚点分布 `p_θ(i|S) = softmax((s_i - s_anchor_i) / τ)` 而不是 GRPO 的 PPO 风格裁剪。

**本实现使用 on-policy 模式（内存高效）**：`old_log_prob` 作为锚点，无需维护额外的锚点模型。

主要特性：
- **On-Policy 锚点策略**：使用 old_log_prob 作为锚点，内存高效
- **自适应温度 (Adaptive Temperature)**：基于熵的温度调节
- **组合奖励函数**：支持 good_accuracy 等多种奖励

## 快速开始

### 1. 基本训练 (On-Policy 模式)

```bash
cd verlm
python -m verl.trainer.main_adpo \
    algorithm.adv_estimator=adpo \
    algorithm.tau=0.8 \
    algorithm.num_generations=8
```

### 2. 启用自适应温度

```bash
python -m verl.trainer.main_adpo \
    algorithm.use_adaptive_tau=true \
    algorithm.adaptive_tau_alpha=0.5 \
    algorithm.adaptive_tau_min=0.05
```

### 3. 使用 good_accuracy 奖励

```bash
python -m verl.trainer.main_adpo \
    custom_reward_function.path=verl/trainer/adpo/reward.py \
    custom_reward_function.name=good_accuracy
```

## 配置参数

### ADPO 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau` | 0.8 | 锚点 softmax 温度 |
| `num_generations` | 8 | 每个 prompt 的生成数量 |
| `beta_reward` | 0.5 | 奖励 softmax 温度 |

### 自适应温度参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_adaptive_tau` | True | 是否使用自适应温度 |
| `adaptive_tau_alpha` | 0.5 | 调制强度 |
| `adaptive_tau_min` | 0.05 | 最小温度值 |

### 损失函数参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beta_anchor_kl` | 0.0 | 锚点 KL 惩罚系数 |
| `drop_all_failed_prompts` | False | 是否丢弃全部失败的 prompt |
| `scale_rewards` | group | 奖励缩放方式: batch/group/none |

## 奖励函数

### 使用 good_accuracy 奖励

**注意**: good_accuracy 使用 VERL 内置的 `prime_math`（基于 sympy），无需额外依赖！

```yaml
# config/adpo_trainer.yaml
custom_reward_function:
  path: verl/trainer/adpo/reward.py
  name: good_accuracy
  
reward_model:
  reward_kwargs:
    ngram_size: 4
    max_penalty: -0.5
    penalty_scale_factor: 0.1
```

或者在代码中直接使用：

```python
from verl.trainer.adpo.reward import good_accuracy

# 在自定义奖励函数中调用
rewards = good_accuracy(
    completions=completions,
    solution=solutions,
    ngram_size=4,
    max_penalty=-0.5,
    penalty_scale_factor=0.1
)
```

### 定义自定义奖励函数

```python
# my_rewards.py
def my_custom_reward(completions, solution, **kwargs):
    """
    自定义奖励函数
    
    Args:
        completions: list[list[dict]] - 模型生成的补全
        solution: list[str] - 参考答案
        **kwargs: 其他数据集列
        
    Returns:
        list[float] - 奖励列表
    """
    rewards = []
    for comp, sol in zip(completions, solution):
        content = comp[0]["content"]
        # 你的奖励逻辑
        reward = compute_my_reward(content, sol)
        rewards.append(reward)
    return rewards
```

然后在配置中引用：

```yaml
custom_reward_function:
  path: my_rewards.py
  name: my_custom_reward
```

## 完整示例配置

```yaml
# config/my_adpo_config.yaml
defaults:
  - adpo_trainer

algorithm:
  # 基础设置
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
  scale_rewards: group
  
  # 可选：锚点 KL 惩罚
  beta_anchor_kl: 0.0
  
  # 可选：丢弃失败样本
  drop_all_failed_prompts: False

# 自定义奖励
custom_reward_function:
  path: verl/trainer/adpo/reward.py
  name: good_accuracy

reward_model:
  reward_kwargs:
    ngram_size: 4
    max_penalty: -0.5
    penalty_scale_factor: 0.1

trainer:
  project_name: my_adpo_project
  experiment_name: adpo_math_gsm8k
  total_epochs: 30
```

运行：

```bash
python -m verl.trainer.main_adpo --config-name my_adpo_config
```

## 监控指标

ADPO 训练会自动记录以下指标：

- `adpo/anchor_kl`: 当前策略与锚点的 KL 散度
- `adpo/mean_tau`: 平均温度值 (如果使用自适应温度)
- `adpo/loss`: ADPO 损失
- `adpo/kl_penalty`: KL 惩罚值 (如果启用)
- `adpo/dropped_prompts`: 丢弃的 prompt 数量 (如果启用)

## 与 GRPO 的对比

| 特性 | GRPO | ADPO (On-Policy) |
|------|------|------------------|
| 核心机制 | PPO 裁剪 | 锚点分布 |
| 参考分布 | old_log_prob | old_log_prob (锚点) |
| 温度控制 | 固定 | 自适应可选 |
| 内存效率 | 高 | 高 (无额外锚点模型) |
| 几何稳定性 | 中等 | 高 |

## 内存优化提示

On-policy ADPO 的优势：
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

## 常见问题

### 1. 批大小必须是 num_generations 的倍数吗？

推荐是，ADPO 会自动截断不符合的批次。建议设置：
```yaml
trainer:
  per_device_train_batch_size: 16  # 确保是 num_generations (8) 的倍数
```

### 2. tau 参数如何调整？

- **低 tau (0.1-0.5)**：更锐利的分布，强调最优样本
- **中 tau (0.5-1.0)**：平衡探索与利用 (推荐)
- **高 tau (>1.0)**：更平滑的分布，更多探索

### 3. 自适应温度何时有用？

当你的任务有多样化的难度时：
- 简单 prompt → 低熵 → 低 tau (更锐利)
- 困难 prompt → 高熵 → 高 tau (更探索)

## 引用

如果你使用 ADPO，请引用：

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

## 贡献

欢迎提交 Issue 和 Pull Request！

