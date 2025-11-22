# ADPO 快速入门指南

## 1. 验证安装

首先验证 ADPO 是否正确安装：

```bash
cd verlm
python examples/test_adpo_installation.py
```

如果所有测试都通过，你就可以开始使用 ADPO 了！

## 2. 运行第一个 ADPO 训练

### 方法 1: 使用交互式快速开始脚本（推荐）

```bash
bash examples/quickstart_adpo.sh
```

这个脚本会提供一个菜单，让你选择不同的 ADPO 模式。

### 方法 2: 直接使用命令行

```bash
# 基础 ADPO 训练（on-policy 模式）
python -m verl.trainer.main_adpo \
    algorithm.adv_estimator=adpo \
    algorithm.anchor_update_mode=on_policy \
    algorithm.num_generations=8 \
    algorithm.tau=0.8
```

### 方法 3: 使用预定义的示例脚本

```bash
# GSM8K 数据集示例
bash examples/run_adpo_gsm8k.sh

# 固定锚点模式示例
bash examples/run_adpo_fixed_anchor.sh

# EMA 更新模式示例
bash examples/run_adpo_ema.sh
```

## 3. 使用 good_accuracy 奖励函数

### 安装依赖

```bash
pip install latex2sympy2_extended math_verify
```

### 运行训练

```bash
python -m verl.trainer.main_adpo \
    --config-name adpo_trainer \
    algorithm.adv_estimator=adpo \
    custom_reward_function.path=verl/trainer/adpo/reward.py \
    custom_reward_function.name=good_accuracy \
    reward_model.reward_kwargs.ngram_size=4 \
    reward_model.reward_kwargs.max_penalty=-0.5
```

## 4. 自定义配置

### 创建 Python 配置文件

```python
from examples.adpo_example_config import get_adpo_on_policy_config
from verl.trainer.main_adpo import run_adpo
from omegaconf import OmegaConf

# 加载基础配置
config = get_adpo_on_policy_config()

# 自定义配置
with OmegaConf.open_dict(config):
    config.algorithm.num_generations = 16
    config.algorithm.tau = 0.5
    config.trainer.total_epochs = 50

# 运行训练
run_adpo(config)
```

### 创建 YAML 配置文件

```yaml
# my_adpo_config.yaml
defaults:
  - adpo_trainer

algorithm:
  adv_estimator: adpo
  num_generations: 8
  tau: 0.8
  anchor_update_mode: on_policy
  use_adaptive_tau: True

trainer:
  project_name: my_project
  experiment_name: my_adpo_experiment
  total_epochs: 30
```

然后运行：

```bash
python -m verl.trainer.main_adpo --config-name my_adpo_config
```

## 5. 主要配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm.tau` | 0.8 | 锚点 softmax 温度 |
| `algorithm.anchor_update_mode` | on_policy | 锚点更新模式 |
| `algorithm.num_generations` | 8 | 每个 prompt 的生成数 |
| `algorithm.use_adaptive_tau` | True | 是否使用自适应温度 |
| `algorithm.beta_reward` | 0.5 | 奖励 softmax 温度 |

## 6. 选择锚点更新模式

```bash
# On-policy（推荐用于快速实验）
python -m verl.trainer.main_adpo algorithm.anchor_update_mode=on_policy

# Fixed（推荐用于最大稳定性）
python -m verl.trainer.main_adpo algorithm.anchor_update_mode=fixed

# EMA（推荐用于平滑演化）
python -m verl.trainer.main_adpo \
    algorithm.anchor_update_mode=ema \
    algorithm.ema_alpha=0.99

# KL-triggered（推荐用于自适应场景）
python -m verl.trainer.main_adpo \
    algorithm.anchor_update_mode=kl_triggered \
    algorithm.kl_threshold=0.1
```

## 7. 监控训练

### 查看日志

```bash
# 训练过程会输出到终端
# 日志也会保存到输出目录
```

### 使用 WandB（如果配置）

```bash
# 在配置中启用 wandb
trainer.logger=["console", "wandb"]
```

### 关键指标

- `adpo/anchor_kl`: 策略与锚点的 KL 散度
- `adpo/mean_tau`: 平均温度值
- `adpo/loss`: ADPO 损失
- `reward`: 平均奖励

## 8. 常见问题排查

### 问题 1: 批大小警告

```
Warning: per_device_train_batch_size (12) is not divisible by num_generations (8)
```

**解决**: 设置批大小为 `num_generations` 的倍数

```bash
python -m verl.trainer.main_adpo \
    algorithm.num_generations=8 \
    trainer.per_device_train_batch_size=16  # 16 = 8 * 2
```

### 问题 2: good_accuracy 导入错误

```
ImportError: No module named 'latex2sympy2_extended'
```

**解决**: 安装依赖

```bash
pip install latex2sympy2_extended math_verify
```

### 问题 3: Ray 初始化失败

```
RuntimeError: Ray is not initialized
```

**解决**: 确保 Ray 配置正确

```yaml
ray_kwargs:
  ray_init:
    num_cpus: null  # 自动检测
```

## 9. 下一步

- 📖 **详细文档**: `verl/trainer/adpo/README.md`
- 🔧 **配置示例**: `examples/adpo_example_config.py`
- 📊 **集成总结**: `ADPO_INTEGRATION_SUMMARY.md`

## 10. 获取帮助

1. 阅读 README: `verl/trainer/adpo/README.md`
2. 查看示例脚本: `examples/run_adpo_*.sh`
3. 运行测试: `python examples/test_adpo_installation.py`
4. 提交 Issue

---

**祝你训练顺利！** 🚀

