# verl 框架 ADPO/GRPO/PPO 训练 Batch Size 参数完全指南

> 本文档详细解释训练脚本中各参数的含义、相互关系、约束条件，以及对显存和速度的影响。

---

## 📊 参数一览表

| 参数 | 默认值 | 显存影响 | 速度影响 | 说明 |
|------|--------|----------|----------|------|
| `train_batch_size` | 1024 | ⭐ | ⭐⭐⭐⭐ | 每步处理的 prompt 数量 |
| `rollout.n` | 1 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 每个 prompt 生成的 response 数量 |
| `ppo_mini_batch_size` | 256 | ⭐⭐⭐ | ⭐⭐⭐ | PPO 更新时的全局 mini-batch 大小 |
| `ppo_micro_batch_size_per_gpu` | null | ⭐⭐ | ⭐⭐ | 单 GPU forward/backward 批大小 |
| `log_prob_micro_batch_size_per_gpu` | null | ⭐⭐ | ⭐ | log prob 计算时单 GPU 批大小 |
| `gpu_memory_utilization` | 0.5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | vLLM KV cache 预分配比例 |
| `max_num_seqs` | 1024 | ⭐⭐ | ⭐⭐ | vLLM 最大并发序列数 |
| `max_prompt_length` | - | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 最大 prompt 长度 |
| `max_response_length` | - | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 最大 response 长度 |

---

## 🔗 参数关系图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           数据流                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   train_batch_size (64 prompts)                                     │
│          │                                                          │
│          ↓                                                          │
│   ┌──────┴──────┐                                                   │
│   │  × rollout.n (8)  │  ← 每个 prompt 生成 n 个 response           │
│   └──────┬──────┘                                                   │
│          ↓                                                          │
│   real_train_batch_size = 64 × 8 = 512 responses                    │
│          │                                                          │
│          ↓                                                          │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │              Rollout 阶段 (vLLM/SGLang)                  │      │
│   │  gpu_memory_utilization=0.35 → KV cache 预分配           │      │
│   │  max_num_seqs=192 → 最大并发数                           │      │
│   │  log_prob_micro_batch_size_per_gpu=16 → log prob 批次    │      │
│   └──────────────────────────────────────────────────────────┘      │
│          │                                                          │
│          ↓                                                          │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │              Training 阶段 (Actor Update)                │      │
│   │                                                          │      │
│   │  ppo_mini_batch_size=32                                  │      │
│   │       ↓ (× rollout.n / n_gpus)                          │      │
│   │  normalized_mini_batch = 32 × 8 / 4 = 64 per GPU        │      │
│   │       ↓                                                  │      │
│   │  ppo_micro_batch_size_per_gpu=8                          │      │
│   │       ↓                                                  │      │
│   │  gradient_accumulation_steps = 64 / 8 = 8 步             │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📝 参数详解

### 1. `data.train_batch_size`

**定义**：每个训练步骤处理的 **prompt 数量**。

**影响**：
- ⬆ 增大：每步处理更多数据，总步数减少，但显存峰值略增
- ⬇ 减小：每步处理更少数据，总步数增加，显存降低

**计算关系**：
```python
real_train_batch_size = train_batch_size × rollout.n
# 这是实际的 response 总数
```

---

### 2. `actor_rollout_ref.rollout.n`

**定义**：每个 prompt 生成的 response 数量（采样次数）。

**关键约束**：
- **GRPO/ADPO 必须 > 1**（需要组内对比）
- PPO 通常 = 1

**影响**：
- ⬆ 增大：生成更多样本，rollout 阶段显存和时间线性增加
- ⬇ 减小：样本多样性降低

**与其他参数的关系**：
```python
# 总样本数
total_responses = train_batch_size × rollout.n

# normalized_mini_batch_size 计算
normalized_mini_batch = ppo_mini_batch_size × rollout.n / n_gpus
```

---

### 3. `actor_rollout_ref.actor.ppo_mini_batch_size`

**定义**：PPO 更新时的**全局** mini-batch 大小（response 数量）。

**约束条件**：
```python
# 必须满足
ppo_mini_batch_size ≤ train_batch_size

# Worker 内部归一化
normalized = ppo_mini_batch_size × rollout.n / n_gpus
# 必须满足
normalized % ppo_micro_batch_size_per_gpu == 0
```

**影响**：
- ⬆ 增大：每次更新使用更多样本，梯度更稳定
- ⬇ 减小：更频繁更新，可能更快收敛但噪声更大

---

### 4. `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`

**定义**：单 GPU 上每次 forward/backward 的样本数。用于**梯度累积**。

**约束条件**：
```python
# 必须整除 normalized_mini_batch_size
normalized_mini_batch % ppo_micro_batch_size_per_gpu == 0
```

**梯度累积步数计算**：
```python
gradient_accumulation_steps = normalized_mini_batch / ppo_micro_batch_size_per_gpu
```

**影响**：
- ⬆ 增大：每次处理更多，训练更快，但显存峰值更高
- ⬇ 减小：显存降低，但梯度累积步数增加，训练变慢

**示例**：
| ppo_micro_batch_size_per_gpu | normalized_mini_batch | 梯度累积步数 | 显存 | 速度 |
|------------------------------|----------------------|--------------|------|------|
| 8 | 64 | 8 | 高 | 快 |
| 4 | 64 | 16 | 中 | 中 |
| 2 | 64 | 32 | 低 | 慢 |

---

### 5. `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`

**定义**：计算 old log probability 时的单 GPU 批大小。

**无严格整除约束**，可自由调整。

**影响**：
- 仅影响 log prob 计算阶段的显存
- 对训练速度影响较小（通常 log prob 计算很快）

---

### 6. `actor_rollout_ref.rollout.gpu_memory_utilization`

**定义**：vLLM 引擎预分配 GPU 显存的比例。

**取值范围**：0.0 ~ 1.0（推荐 0.35 ~ 0.7）

**影响**：
- ⬆ 增大：更大的 KV cache，rollout 更快，但可能挤占训练显存
- ⬇ 减小：rollout 变慢，但给训练阶段留更多显存

**重要**：此参数只影响 **rollout 阶段**，设置 `free_cache_engine=True` 后训练前会释放。

---

### 7. `actor_rollout_ref.rollout.max_num_seqs`

**定义**：vLLM 引擎同时处理的最大序列数。

**影响**：
- ⬆ 增大：更高并发，更好的 GPU 利用率
- ⬇ 减小：降低并发，可能降低吞吐

---

## ⚖️ 约束条件汇总

### 核心约束公式

```python
# 1. 总样本数必须能被 GPU 数整除
(train_batch_size × rollout.n) % n_gpus == 0

# 2. normalized_mini_batch 必须能被 micro_batch 整除
normalized_mini_batch = ppo_mini_batch_size × rollout.n / n_gpus
normalized_mini_batch % ppo_micro_batch_size_per_gpu == 0

# 3. mini_batch_size 不能超过 train_batch_size
ppo_mini_batch_size ≤ train_batch_size
```

### 4 GPU 配置验证示例

当前配置：
- `train_batch_size=64`
- `rollout.n=8`
- `ppo_mini_batch_size=32`
- `ppo_micro_batch_size_per_gpu=8`
- `n_gpus=4`

验证：
```python
# 约束 1: (64 × 8) % 4 == 512 % 4 == 0 ✅
# 约束 2: (32 × 8 / 4) % 8 == 64 % 8 == 0 ✅
# 约束 3: 32 ≤ 64 ✅

# 梯度累积步数: 64 / 8 = 8 步
```

---

## 🎛️ 调参指南

### 显存不够？按优先级调整

1. **`max_response_length`** - 降到 1024 或更低（影响最大）
2. **`gpu_memory_utilization`** - 降到 0.3~0.4
3. **`ppo_micro_batch_size_per_gpu`** - 降到 4 或 2
4. **`log_prob_micro_batch_size_per_gpu`** - 降到 8 或 4
5. **`train_batch_size`** 和 **`ppo_mini_batch_size`** - 同比例降低

### 速度太慢？按优先级调整

1. **`ppo_micro_batch_size_per_gpu`** - 提高（需要显存支持）
2. **`gpu_memory_utilization`** - 提高到 0.5~0.6
3. **`log_prob_micro_batch_size_per_gpu`** - 提高到 16~32
4. **`max_num_seqs`** - 提高并发数

---
max_response_length=1280	⭐⭐⭐⭐⭐	⭐⭐⭐⭐⭐	影响最大！序列长度直接影响 KV Cache 和激活显存
max_prompt_length=880	⭐⭐⭐⭐	⭐⭐⭐⭐	与 response_length 类似，但通常较短
gpu_memory_utilization=0.45	⭐⭐⭐⭐	⭐⭐⭐	vLLM KV Cache 预分配比例，直接控制 rollout 显存
rollout.n=8	⭐⭐⭐⭐	⭐⭐⭐⭐	每个 prompt 生成 n 个响应，显存和计算量线性增长
train_batch_size=128	⭐⭐⭐	⭐⭐⭐⭐	总体批次大小，通过梯度累积分摊
ppo_mini_batch_size=64	⭐⭐⭐	⭐⭐⭐	每次更新的样本数，影响梯度累积步数
ppo_micro_batch_size_per_gpu=8	⭐⭐	⭐⭐	单 GPU 前向/反向批次，降低可减少训练峰值显存
log_prob_micro_batch_size_per_gpu=8	⭐⭐	⭐	log prob 计算批次，仅影响该阶段显存
max_num_seqs=192	⭐⭐	⭐⭐	vLLM 并发序列数，影响 rollout 调度
fsdp_config.param_offload=False	⭐⭐⭐	⭐⭐⭐	开启可大幅降低显存，但显著降速
enforce_eager=False	⭐	⭐⭐	True 禁用 CUDA Graph，降显存但降速
enable_chunked_prefill=True	⭐	⭐	分块 prefill，略微降显存
enable_prefix_caching=True	⭐	⭐⭐	前缀缓存，可能加速重复 prompt
free_cache_engine=True	⭐⭐	⭐	rollout 后释放 KV Cache，降显存但有重建开销
## 🔄 ADPO vs GRPO vs PPO 对比

| 特性 | PPO | GRPO | ADPO |
|------|-----|------|------|
| **Critic 模型** | ✅ 需要 | ❌ 无需 | ❌ 无需 |
| **显存开销** | 高（双模型） | 低 | 低 |
| **rollout.n 要求** | 通常 = 1 | **> 1** | **> 1** |
| **优势估计器** | GAE | Group-relative | Anchored softmax |

### GRPO/ADPO 特殊要求

```yaml
# rollout.n 必须 > 1，用于组内对比
actor_rollout_ref:
  rollout:
    n: 8  # 必须 > 1！
    
algorithm:
  adv_estimator: grpo  # 或 adpo
  num_generations: 8   # ADPO 需要，通常等于 rollout.n
```

---

## 📊 实际配置示例

### 4x4090 (24GB) ADPO 配置

```bash
# 保守配置（稳定运行）
data.train_batch_size=64
data.max_prompt_length=880
data.max_response_length=1280
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.gpu_memory_utilization=0.35
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8

# 计算验证
# real_batch = 64 × 8 = 512, 512 % 4 = 0 ✅
# normalized_mini = 32 × 8 / 4 = 64, 64 % 8 = 0 ✅
# grad_accum = 64 / 8 = 8 步
```

### 8xA100 (80GB) GRPO 高速配置

```bash
data.train_batch_size=256
data.max_prompt_length=1024
data.max_response_length=2048
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32
actor_rollout_ref.actor.ppo_mini_batch_size=128
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16

# 计算验证
# real_batch = 256 × 8 = 2048, 2048 % 8 = 0 ✅
# normalized_mini = 128 × 8 / 8 = 128, 128 % 16 = 0 ✅
# grad_accum = 128 / 16 = 8 步
```

---

## ❓ 常见问题

### Q1: 报错 `normalized ppo_mini_batch_size should be divisible by ppo_micro_batch_size_per_gpu`

**原因**：约束 2 不满足。

**解决**：调整 `ppo_micro_batch_size_per_gpu` 使其能整除 `normalized_mini_batch`。

```python
normalized = ppo_mini_batch_size × rollout.n / n_gpus
# 选择能整除 normalized 的值
```

### Q2: OOM 在 rollout 阶段

**解决**：
1. 降低 `gpu_memory_utilization` (如 0.3)
2. 降低 `max_num_seqs` (如 64)
3. 减小 `max_response_length`

### Q3: OOM 在 training 阶段

**解决**：
1. 降低 `ppo_micro_batch_size_per_gpu` (如 2 或 1)
2. 开启 `gradient_checkpointing: true`
3. 开启 `fsdp_config.param_offload: true` (会变慢)

### Q4: 训练速度很慢

**检查**：
1. `ppo_micro_batch_size_per_gpu` 是否太小（增加梯度累积步数）
2. `gpu_memory_utilization` 是否太低
3. 是否开启了 `param_offload`（应该关闭）

---

## 📈 性能监控指标

运行时关注这些 timing 指标：

| 指标 | 说明 | 优化方向 |
|------|------|----------|
| `timing_s/gen` | Rollout 生成时间 | 提高 `gpu_memory_utilization`, `max_num_seqs` |
| `timing_s/update_actor` | Actor 更新时间 | 提高 `ppo_micro_batch_size_per_gpu` |
| `timing_s/old_log_prob` | Log prob 计算时间 | 提高 `log_prob_micro_batch_size_per_gpu` |
| `timing_s/reward` | Reward 计算时间 | 通常很快，无需优化 |
| `perf/throughput` | Token 吞吐量 | 综合指标，越高越好 |

---

## 🔧 显存优化特性汇总

verl 提供了多种显存优化特性，以下按**推荐程度**排序：

### 1. Gradient Checkpointing (梯度检查点) ⭐⭐⭐⭐⭐

**效果**：显存降低 30-50%，速度降低 10-20%

```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: True
```

**原理**：训练时不保存所有中间激活值，反向传播时重新计算，用时间换显存。

---

### 2. Sequence Packing (序列打包) ⭐⭐⭐⭐⭐

**效果**：提高 GPU 利用率，减少 padding 浪费

```yaml
actor_rollout_ref:
  model:
    use_remove_padding: True
```

**支持模型**：Qwen、LLaMA、Mistral、Gemma 等

---

### 3. free_cache_engine (释放 KV Cache) ⭐⭐⭐⭐

**效果**：训练时释放 rollout 阶段的 KV Cache，为训练腾出显存

```yaml
actor_rollout_ref:
  rollout:
    free_cache_engine: True
```

**注意**：下次 rollout 需要重新预热，有少量开销。

---

### 4. FSDP2 (新一代分布式训练) ⭐⭐⭐⭐

**效果**：比 FSDP1 显存降低 7%，吞吐提升 1.5%

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp2
  ref:
    strategy: fsdp2
```

**要求**：PyTorch 2.1+

---

### 5. Activation Offload (激活值卸载) ⭐⭐⭐

**效果**：将激活值卸载到 CPU，配合 gradient checkpointing 使用

```yaml
actor_rollout_ref:
  model:
    enable_activation_offload: True
    enable_gradient_checkpointing: True  # 必须一起开启
```

**注意**：仅 FSDP 后端支持，会降低速度。

---

### 6. CPU Offload (参数/优化器卸载) ⭐⭐⭐

**效果**：大幅降低显存，但**显著降低训练速度**

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      param_offload: True       # 参数卸载到 CPU
      optimizer_offload: True   # 优化器状态卸载到 CPU
```

**FSDP2 专属**：
```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      offload_policy: True  # FSDP2 的 CPU offload，兼容梯度累积
```

**⚠️ 警告**：这是最后手段，速度会明显变慢！

---

### 7. Entropy Chunking (熵计算分块) ⭐⭐

**效果**：降低 logits 的显存峰值

```yaml
actor_rollout_ref:
  ref:
    entropy_from_logits_with_chunking: True
  actor:
    entropy_checkpointing: True  # 训练时的熵重计算
```

---

### 8. Dynamic Batch Size (动态批大小) ⭐⭐

**效果**：按 token 数而非样本数分批，减少显存浪费

```yaml
actor_rollout_ref:
  actor:
    use_dynamic_bsz: True
    ppo_max_token_len_per_gpu: 8192  # 替代 micro_batch_size
```

---

### 9. Liger Kernel (高性能内核) ⭐⭐

**效果**：SFT 训练效率提升，显存略降

```yaml
model:
  use_liger: True
```

**安装**：`pip install liger-kernel`

---

### 10. FP8 Rollout ⭐

**效果**：使用 FP8 进行推理，降低 rollout 显存

需要 Hopper 架构 GPU (H100/H200)。

---

## 📋 显存优化配置模板

### 极限省显存配置（4x4090 小模型）

```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: True
    use_remove_padding: True
    enable_activation_offload: True
  actor:
    ppo_micro_batch_size_per_gpu: 2
    fsdp_config:
      param_offload: False  # 不建议开，太慢
  rollout:
    gpu_memory_utilization: 0.35
    free_cache_engine: True
    log_prob_micro_batch_size_per_gpu: 8
```

### 平衡配置（推荐）

```yaml
actor_rollout_ref:
  model:
    enable_gradient_checkpointing: True
    use_remove_padding: True
  actor:
    ppo_micro_batch_size_per_gpu: 8
    fsdp_config:
      param_offload: False
  rollout:
    gpu_memory_utilization: 0.5
    free_cache_engine: True
    log_prob_micro_batch_size_per_gpu: 16
```

---

*文档版本: 2025-11-29*
