# ADPO (Anchored Direct Preference Optimization) Trainer

ADPO 是一种新的强化学习算法，通过锚点分布提供几何稳定性来优化策略。

## 核心思想

ADPO 使用锚点分布 `p_θ(i|S) = softmax((s_i - s_anchor_i) / τ)` 而不是 GRPO 的 PPO 风格裁剪。

主要特性：
- **锚点策略 (Anchor Policy)**：提供稳定的参考分布
- **自适应温度 (Adaptive Temperature)**：基于熵的温度调节
- **多种更新模式**：on-policy、fixed、EMA、KL-triggered
- **组合奖励函数**：支持 good_accuracy 等多种奖励

## 快速开始

### 1. 基本训练

```bash
cd verlm
python -m verl.trainer.main_adpo \
    algorithm.adv_estimator=adpo \
    algorithm.tau=0.8 \
    algorithm.anchor_update_mode=on_policy \
    algorithm.num_generations=8
```

### 2. 使用固定锚点 (标准 ADPO)

```bash
python -m verl.trainer.main_adpo \
    algorithm.anchor_update_mode=fixed \
    algorithm.tau=1.0
```

### 3. 使用 EMA 动态锚点

```bash
python -m verl.trainer.main_adpo \
    algorithm.anchor_update_mode=ema \
    algorithm.ema_alpha=0.99 \
    algorithm.tau=0.8
```

### 4. 使用 KL 触发更新

```bash
python -m verl.trainer.main_adpo \
    algorithm.anchor_update_mode=kl_triggered \
    algorithm.kl_threshold=0.1
```

## 配置参数

### ADPO 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau` | 0.8 | 锚点 softmax 温度 |
| `anchor_update_mode` | on_policy | 锚点更新模式: on_policy/fixed/ema/kl_triggered |
| `num_generations` | 8 | 每个 prompt 的生成数量 |

### 自适应温度参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_adaptive_tau` | True | 是否使用自适应温度 |
| `adaptive_tau_alpha` | 0.5 | 调制强度 |
| `adaptive_tau_min` | 0.05 | 最小温度值 |

### 锚点更新参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ema_alpha` | 0.99 | EMA 系数 (仅用于 ema 模式) |
| `kl_threshold` | 0.1 | KL 阈值 (仅用于 kl_triggered 模式) |

### 损失函数参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beta_anchor_kl` | 0.0 | 锚点 KL 惩罚系数 |
| `beta_reward` | 0.5 | 奖励 softmax 温度 |
| `use_q_centering` | True | 是否中心化优势 |
| `drop_all_failed_prompts` | False | 是否丢弃全部失败的 prompt |

## 奖励函数

### 使用 good_accuracy 奖励

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

## 锚点更新模式详解

### 1. On-Policy (默认)
- 类似 GRPO，使用 `old_per_token_logps` 作为锚点
- 锚点在每次生成时更新
- 适合快速收敛场景

```yaml
algorithm:
  anchor_update_mode: on_policy
```

### 2. Fixed (标准 ADPO)
- 锚点固定为初始策略，从不更新
- 真正的离线 ADPO
- 适合稳定性优先场景

```yaml
algorithm:
  anchor_update_mode: fixed
```

### 3. EMA (指数移动平均)
- 每步更新：`anchor = alpha * anchor + (1-alpha) * current`
- 平滑的策略演化
- 适合需要渐进更新的场景

```yaml
algorithm:
  anchor_update_mode: ema
  ema_alpha: 0.99  # 更高 = 更稳定
```

### 4. KL-Triggered (KL 触发)
- 当 KL 散度超过阈值时硬更新
- 自适应更新频率
- 适合动态平衡探索与利用

```yaml
algorithm:
  anchor_update_mode: kl_triggered
  kl_threshold: 0.1
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
  
  # ADPO 核心参数
  tau: 0.8
  anchor_update_mode: on_policy
  
  # 自适应温度
  use_adaptive_tau: True
  adaptive_tau_alpha: 0.5
  adaptive_tau_min: 0.05
  
  # 奖励计算
  beta_reward: 0.5
  use_q_centering: True
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
- `adpo/anchor_updates`: 锚点更新次数
- `adpo/mean_tau`: 平均温度值
- `adpo/loss`: ADPO 损失
- `adpo/kl_penalty`: KL 惩罚值 (如果启用)
- `adpo/dropped_prompts`: 丢弃的 prompt 数量 (如果启用)

## 与 GRPO 的对比

| 特性 | GRPO | ADPO |
|------|------|------|
| 核心机制 | PPO 裁剪 | 锚点分布 |
| 参考分布 | old_log_prob | 锚点策略 |
| 温度控制 | 固定 | 自适应可选 |
| 更新灵活性 | 固定频率 | 多种模式 |
| 几何稳定性 | 中等 | 高 |

## 常见问题

### 1. 批大小必须是 num_generations 的倍数吗？

是的，ADPO 会自动截断不符合的批次。建议设置：
```yaml
trainer:
  per_device_train_batch_size: 16  # 确保是 num_generations (8) 的倍数
```

### 2. 如何选择锚点更新模式？

- **快速实验**：使用 `on_policy` (类似 GRPO)
- **最大稳定性**：使用 `fixed`
- **平滑演化**：使用 `ema` (alpha=0.99)
- **自适应**：使用 `kl_triggered` (threshold=0.1)

### 3. tau 参数如何调整？

- **低 tau (0.1-0.5)**：更锐利的分布，强调最优样本
- **中 tau (0.5-1.0)**：平衡探索与利用 (推荐)
- **高 tau (>1.0)**：更平滑的分布，更多探索

### 4. 自适应温度何时有用？

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

