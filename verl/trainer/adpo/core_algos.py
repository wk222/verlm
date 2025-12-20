# Copyright 2025 ADPO Algorithm Author
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ADPO (Anchored Direct Preference Optimization) - 高效向量化实现
"""
ADPO核心算法 - 保留完整功能的向量化实现

功能：
- 多种loss变体（plackett_luce, softmax, scaled, direct）
- 自适应温度策略
- 梯度裁剪
- 完全向量化（无Python循环）
- Index Mode 支持任意 batch size
"""

from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from verl.utils import as_torch_index
from verl.trainer.ppo.core_algos import register_adv_est, register_policy_loss


# ============================================================
# 延迟 Softmax CE：支持跨 micro-batch 的组计算
# ============================================================

class DelayedGroupSoftmaxCE(torch.autograd.Function):
    """
    支持组成员分布在不同 micro-batch 的 Softmax CE Loss
    
    核心思想：
    - 前向时只计算 -q·u（不需要组内完整）
    - 梯度使用预计算的 p = softmax(u_group)
    
    数学：
    - 标准 CE: L = -q·u + logsumexp(u)
    - 梯度: ∂L/∂u = softmax(u) - q = p - q
    
    关键：只要我们能提供正确的 p，就能得到正确的梯度，
          而 p 可以在所有组成员收集完后统一计算
    """
    
    @staticmethod
    def forward(ctx, u, q, p_precomputed):
        """
        Args:
            u: (P, G) - 当前 batch 的 anchored scores（需要梯度）
            q: (P, G) - 目标分布（已 detach）
            p_precomputed: (P, G) - 预计算的 softmax(u_全组)
        """
        ctx.save_for_backward(p_precomputed.detach(), q.detach())
        # 用预计算的 p 计算 CE loss（用于 logging）
        eps = 1e-10
        loss = -(q * torch.log(p_precomputed + eps)).sum(dim=-1).mean()
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        p_precomputed, q = ctx.saved_tensors
        # 正确的梯度：p - q
        grad_u = grad_output * (p_precomputed - q)
        return grad_u, None, None


def compute_group_softmax_from_rewards(
    sequence_rewards: torch.Tensor,
    index: np.ndarray,
    beta_reward: float = 0.3,
    tau: float = 0.5,
) -> torch.Tensor:
    """
    根据 reward 预计算 p = softmax(advantage / tau)
    
    这个函数应该在 advantage 计算阶段调用，此时所有组成员的 reward 都已收集完毕
    
    Args:
        sequence_rewards: (B,) 所有样本的序列奖励
        index: (B,) 组索引
        beta_reward: reward softmax 温度
        tau: anchored score 温度
        
    Returns:
        p_target: (B,) 每个样本的目标概率 p_i = exp(adv_i/tau) / sum_j(exp(adv_j/tau))
    """
    device = sequence_rewards.device
    index_tensor = as_torch_index(index, device=device)
    
    # 计算组内均值
    num_groups = index_tensor.max().item() + 1
    group_sum = torch.zeros(num_groups, device=device, dtype=sequence_rewards.dtype)
    group_count = torch.zeros(num_groups, device=device, dtype=sequence_rewards.dtype)
    
    group_sum.index_add_(0, index_tensor, sequence_rewards)
    group_count.index_add_(0, index_tensor, torch.ones_like(sequence_rewards))
    group_count = group_count.clamp(min=1.0)
    group_mean = group_sum / group_count
    
    # 计算 advantage
    advantages = sequence_rewards - group_mean[index_tensor]
    
    # 计算 p = softmax(advantage / beta_reward)
    # 使用 scatter 实现组内 softmax
    scaled_adv = advantages / beta_reward
    
    # 计算组内 logsumexp（用于归一化）
    # 先计算组内 max（数值稳定）
    group_max = torch.full((num_groups,), float('-inf'), device=device, dtype=sequence_rewards.dtype)
    group_max = torch.scatter_reduce(
        group_max, 0, index_tensor, scaled_adv, reduce='amax'
    )
    scaled_adv_shifted = scaled_adv - group_max[index_tensor]
    
    # 计算 exp 和组内 sum
    exp_adv = torch.exp(scaled_adv_shifted)
    group_exp_sum = torch.zeros(num_groups, device=device, dtype=sequence_rewards.dtype)
    group_exp_sum.index_add_(0, index_tensor, exp_adv)
    
    # p = exp(adv - max) / sum(exp(adv - max))
    p_target = exp_adv / group_exp_sum[index_tensor]
    
    return p_target


def _scatter_group_mean_std(values: torch.Tensor, index: torch.Tensor, return_std: bool = False):
    """
    Compute mean and std of values grouped by index.
    Returns tensors of same shape as values, broadcasted back to original indices.
    """
    # Ensure index is LongTensor
    index = index.long()
    num_groups = index.max().item() + 1
    
    # Compute sum and count
    # We use a flattened view for simpler index_add if values has extra dims
    # But here values is usually (B,)
    
    out_sum = torch.zeros(num_groups, device=values.device, dtype=values.dtype)
    out_count = torch.zeros(num_groups, device=values.device, dtype=values.dtype)
    
    out_sum.index_add_(0, index, values)
    out_count.index_add_(0, index, torch.ones_like(values))
    
    # Avoid division by zero
    out_count = out_count.clamp(min=1.0)
    group_mean = out_sum / out_count
    
    broadcast_mean = group_mean[index]
    
    if not return_std:
        return broadcast_mean, None
        
    # Compute std with Bessel's correction (unbiased)
    out_sq_sum = torch.zeros(num_groups, device=values.device, dtype=values.dtype)
    out_sq_sum.index_add_(0, index, values.pow(2))
    
    # Var = (Sum(x^2) - (Sum x)^2 / N) / (N - 1)
    numerator = out_sq_sum - (out_sum.pow(2) / out_count)
    numerator = numerator.clamp(min=0.0) # Avoid negative due to precision
    denominator = (out_count - 1.0).clamp(min=1.0)
    group_var = numerator / denominator
    
    group_std = group_var.clamp(min=1e-10).sqrt()
    
    # Handle singleton groups (count <= 1)
    # For singletons, std is technically undefined or 0. We set it to 1.0 to avoid division by zero/tiny.
    is_singleton = out_count <= 1.0
    if is_singleton.any():
        group_std = torch.where(is_singleton, torch.ones_like(group_std), group_std)
    
    broadcast_std = group_std[index]
    
    return broadcast_mean, broadcast_std


@register_adv_est("adpo")
def compute_adpo_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: DictConfig,
    index: Optional[np.ndarray] = None,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ADPO advantages (Sequence Level, returns (B,) dim)
    
    Fully vectorized implementation supporting arbitrary batch sizes.
    
    当启用 use_delayed_softmax 时，还会返回预计算的 p_target，
    通过 kwargs 中的 extra_returns dict 返回。
    """
    with torch.no_grad():
        sequence_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # (B,)
        
        batch_size = sequence_rewards.shape[0]
        device = sequence_rewards.device
        num_generations = config.get("num_generations", 8) if config else 8
        scale_rewards = config.get("scale_rewards", True) if config else True
        beta_reward = config.get("beta_reward", 0.3) if config else 0.3
        use_delayed_softmax = config.get("use_delayed_softmax", False) if config else False
        
        # Convert index to tensor if present
        if index is not None:
            # Use as_torch_index to handle various index types (int, str/uuid, etc.)
            # and map them to contiguous range [0, G-1]
            index_tensor = as_torch_index(index, device=device)
                
            # Vectorized Group Mean/Std
            mean_rewards, std_rewards = _scatter_group_mean_std(sequence_rewards, index_tensor, return_std=scale_rewards)
            
            if scale_rewards:
                # std_rewards is already broadcasted to (B,)
                std_rewards = torch.where(std_rewards < 1e-8, torch.ones_like(std_rewards), std_rewards)
                advantages = (sequence_rewards - mean_rewards) / std_rewards
            else:
                advantages = sequence_rewards - mean_rewards
            
            # ============================================================
            # 延迟 Softmax 模式：预计算 p_target = softmax(advantage / beta)
            # 
            # 这里是最佳时机，因为：
            # 1. 所有组成员的 reward 已经收集完毕
            # 2. 可以正确计算组内 softmax
            # 3. 后续 loss 计算时可以跨 micro-batch
            # ============================================================
            if use_delayed_softmax or config.get("use_precomputed_q", False):
                # 计算 p_target = softmax(advantage / beta_reward)，按组归一化
                # 使用 scatter 实现组内 softmax
                scaled_adv = advantages / beta_reward
                
                # 数值稳定：先减去组内 max
                num_groups = index_tensor.max().item() + 1
                group_max = torch.full((num_groups,), float('-inf'), device=device, dtype=sequence_rewards.dtype)
                group_max = torch.scatter_reduce(group_max, 0, index_tensor, scaled_adv, reduce='amax')
                scaled_adv_shifted = scaled_adv - group_max[index_tensor]
                
                # exp 并求组内 sum
                exp_adv = torch.exp(scaled_adv_shifted)
                group_exp_sum = torch.zeros(num_groups, device=device, dtype=sequence_rewards.dtype)
                group_exp_sum.index_add_(0, index_tensor, exp_adv)
                
                # p = exp(adv - max) / sum(exp(adv - max))
                p_target = exp_adv / group_exp_sum[index_tensor]
                
                # 通过 kwargs 返回（调用方需要处理）
                extra_returns = kwargs.get("extra_returns", None)
                if extra_returns is not None and isinstance(extra_returns, dict):
                    extra_returns["q_target"] = p_target
                
        else:
            # Fallback: Global Mean/Std (if no index provided)
            # This is efficient but conceptually treats the whole batch as one group
            mean_reward = sequence_rewards.mean()
            if scale_rewards:
                std_reward = sequence_rewards.std()
                std_reward = std_reward if std_reward > 1e-8 else torch.tensor(1.0, device=device)
                advantages = (sequence_rewards - mean_reward) / std_reward
            else:
                advantages = sequence_rewards - mean_reward
            
            # 全局模式下也支持 p_target 预计算
            if use_delayed_softmax:
                p_target = F.softmax(advantages / beta_reward, dim=-1)
                extra_returns = kwargs.get("extra_returns", None)
                if extra_returns is not None and isinstance(extra_returns, dict):
                    extra_returns["p_target"] = p_target
        
    return advantages, sequence_rewards


def _compute_plackett_luce_loss(
    u: torch.Tensor, 
    adv: torch.Tensor, 
    use_poly_loss: bool = False, 
    epsilon: float = 1.0,
    top_k: int = 0,
    temperature: float = 1.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Plackett-Luce (ListMLE) loss - 完全向量化实现（无循环）
    
    Args:
        u: (P, G) anchored scores
        adv: (P, G) advantages
        use_poly_loss: 是否开启 Poly-Loss 修正
        epsilon: Poly-Loss 的权重系数
        top_k: 只计算前 top_k 个位置的 loss (0 表示全部，推荐 3-5)
        temperature: 对 u 进行温度缩放，>1 使分布更平滑，提高稳定性
        label_smoothing: 标签平滑，软化排序目标，减少对精确排序的依赖
        
    Returns:
        loss: scalar
        
    数学形式:
        Standard: loss = -sum_k [u[σ(k)] - logsumexp(u[σ(k:)])]
        Poly: loss = Standard + epsilon * (1 - p_rank)
        其中 p_rank = exp(-Standard_k)
        
    稳定性改进:
        1. top_k: 只关注前几名，忽略尾部噪声
        2. temperature: 平滑 score 分布
        3. label_smoothing: 软化排序目标
        4. 数值稳定的 cumulative logsumexp
    """
    P, G = u.shape
    
    # 温度缩放 (temperature > 1 使分布更平滑)
    u_scaled = u / temperature
    
    # 按 advantage 排序
    sorted_indices = torch.argsort(adv, dim=-1, descending=True)  # (P, G)
    sorted_u = torch.gather(u_scaled, 1, sorted_indices)  # (P, G)
    
    # 完全向量化的 cumulative logsumexp（从后往前）
    # 使用 torch.logcumsumexp 保证梯度正确传播
    # 注意：旧的 cummax 实现会导致梯度消失问题！
    #   - 在 max 位置：d(x - cummax(x))/dx = 1 - 1 = 0
    #   - 导致 exp(...) 的输入梯度为 0
    sorted_u_flip = torch.flip(sorted_u, dims=[-1])  # (P, G) 翻转
    
    # 使用 PyTorch 原生的 logcumsumexp（梯度正确）
    # logcumsumexp(x)[i] = log(sum(exp(x[0:i+1])))
    cum_logsumexp_flip = torch.logcumsumexp(sorted_u_flip, dim=-1)  # (P, G)
    
    # 翻转回来得到从位置 k 开始的 logsumexp
    # cum_logsumexp[k] = log(sum(exp(sorted_u[k:])))
    cum_logsumexp = torch.flip(cum_logsumexp_flip, dims=[-1])  # (P, G)
    
    # loss_k = -u[k] + logsumexp(u[k:])
    # 这就是 -log(p_rank)
    per_position_loss = -sorted_u[:, :-1] + cum_logsumexp[:, :-1]  # (P, G-1)
    
    # Top-K: 只计算前 k 个位置
    if top_k > 0 and top_k < G - 1:
        per_position_loss = per_position_loss[:, :top_k]  # (P, top_k)
        # Adjust G for subsequent calculations if needed, but here we just slice
    
    # Label Smoothing: 软化排序目标
    # 思想：不要求完美排序，允许一定的误差
    # Correct implementation: L = (1-eps) * L_real + eps * L_uniform
    # L_uniform = KL(Uniform || Model) + Entropy(Uniform)
    # We focus on the CrossEntropy part: H(Uniform, Model) = - sum (1/N) * log p_i
    # = - (1/N) * sum (u_i - logZ) = logZ - (1/N) * sum u_i
    if label_smoothing > 0:
        # Calculate suffix sum of scores for uniform loss
        # sorted_u_flip is already available
        cumsum_u_flip = torch.cumsum(sorted_u_flip, dim=-1)
        cumsum_u = torch.flip(cumsum_u_flip, dims=[-1]) # (P, G)
        
        # Number of elements in each suffix: G, G-1, ..., 1
        suffix_counts = torch.arange(G, 0, -1, device=u.device, dtype=u.dtype) # (G,)
        
        # Mean score of suffix
        suffix_mean_u = cumsum_u / suffix_counts # (P, G)
        
        # Uniform Loss = logZ - mean_u
        # cum_logsumexp is logZ for each position
        uniform_loss = cum_logsumexp - suffix_mean_u # (P, G)
        
        # Slice to match per_position_loss (remove last element and apply top_k)
        uniform_loss = uniform_loss[:, :-1]
        if top_k > 0 and top_k < G - 1:
            uniform_loss = uniform_loss[:, :top_k]
            
        per_position_loss = (1 - label_smoothing) * per_position_loss + label_smoothing * uniform_loss
    
    if use_poly_loss:
        # Poly-Ranking Loss: L = -log(p) + epsilon * (1 - p)
        # per_position_loss is -log(p)
        # 数值稳定：clamp per_position_loss 防止 exp 爆炸
        clamped_loss = torch.clamp(per_position_loss, max=20.0)
        p_rank = torch.exp(-clamped_loss)
        poly_term = epsilon * (1.0 - p_rank)
        per_position_loss = per_position_loss + poly_term
    
    return per_position_loss.mean()





@register_policy_loss("adpo")
def adpo_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "mean",
    config: Optional[DictConfig] = None,
    rollout_log_probs: Optional[torch.Tensor] = None,
    **kwargs
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    ADPO policy loss - 完整功能的向量化实现
    
    Loss变体:
    - "plackett_luce": P-L模型/ListMLE，完整排序对齐 ⭐推荐
    - "softmax": 原始ADPO (支持新公式 u = ℓ - A·ℓ_ref - B·q)
    - "scaled": 原始ADPO × scale
    - "direct": -q·u + logsumexp(u)
    
    功能:
    - 自适应温度策略
    - 梯度裁剪
    - 长度归一化
    - Index Mode 支持任意 batch size（无需是 num_generations 的整数倍）
    """
    assert config is not None, "Config is required for ADPO loss"
    
    # ============================================================
    # 读取配置
    # ============================================================
    def _cfg(key: str, default):
        if hasattr(config, 'policy_loss') and config.policy_loss is not None:
            val = config.policy_loss.get(key, None)
            if val is not None:
                return val
        val = config.get(key, None)
        return val if val is not None else default
    
    # 核心参数
    tau = _cfg("tau", 0.5)
    num_generations = _cfg("num_generations", 8)
    beta_reward = _cfg("beta_reward", 0.3)
    loss_variant = _cfg("loss_variant", "plackett_luce")  # 默认使用 P-L
    clip_anchored_score = _cfg("clip_anchored_score", 10.0)
    use_length_normalization = _cfg("use_length_normalization", True)
    grad_scale_factor = _cfg("grad_scale_factor", 20.0)
    
    # Logit Regularization (Z-Loss) 防止 Logits 爆炸
    logit_reg_coef = _cfg("logit_reg_coef", 0.0)
    
    # Q-Weighted Centering: 消除 Softmax 的零模方向
    use_q_center = _cfg("use_q_center", True)
    
    # 自适应温度参数
    use_adaptive_tau = _cfg("use_adaptive_tau", False)
    adaptive_tau_alpha = _cfg("adaptive_tau_alpha", 0.2)
    adaptive_tau_beta = _cfg("adaptive_tau_beta", 0.5)
    adaptive_tau_min = _cfg("adaptive_tau_min", 0.2)
    adaptive_tau_max = _cfg("adaptive_tau_max", 1.5)
    vocab_size = _cfg("vocab_size", 32000)
    
    # 梯度裁剪
    grad_clip_value = _cfg("grad_clip_value", 0.0)
    clip_log_ratio = _cfg("clip_log_ratio", 5.0)

    # Poly-Loss Config (仅用于 Plackett-Luce Loss)
    # 默认开启 Poly-Loss
    use_poly_loss = _cfg("use_poly_loss", True)
    poly_epsilon = _cfg("poly_epsilon", 1.0)
    
    # Plackett-Luce 稳定性参数
    pl_top_k = _cfg("pl_top_k", 3)  # 只计算前 k 个位置 (0=全部，推荐 3-5)
    pl_temperature = _cfg("pl_temperature", 1.0)  # >1 使分布更平滑
    pl_label_smoothing = _cfg("pl_label_smoothing", 0.1)  # 软化排序目标 (推荐 0.05-0.2)
    
    # Softmax 变体的新公式参数: u = ℓ - A·ℓ_ref - B·q
    # Softmax 变体的超参数: u = (log_prob - A·old_log_prob - B·q - C) / tau
    # 系数 A 控制 reference model 的锚定强度
    # 系数 B 控制 target distribution q 的减法项
    # 系数 C 控制常数偏移项
    # 
    # 变体示例:
    # - Standard ADPO: A=1, B=0, C=0 → u = (log_prob - old_log_prob) / tau
    # - SFT Variant:   A=0, B=0, C=0 → u = log_prob / tau
    # - 其他探索:      A=0.5 等中间值
    softmax_coef_A = _cfg("softmax_coef_A", 1.0)
    softmax_coef_B = _cfg("softmax_coef_B", 0.0)
    softmax_coef_C = _cfg("softmax_coef_C", 0.0)
    
    # ============================================================
    # Step 1: 计算序列级log prob
    # ============================================================
    seq_lengths = response_mask.sum(dim=-1).clamp(min=1)  # (B,)
    
    if use_length_normalization:
        seq_log_prob = (log_prob * response_mask).sum(dim=-1) / seq_lengths
        seq_old_log_prob = (old_log_prob * response_mask).sum(dim=-1) / seq_lengths
    else:
        seq_log_prob = (log_prob * response_mask).sum(dim=-1)
        seq_old_log_prob = (old_log_prob * response_mask).sum(dim=-1)
    
    log_ratio = seq_log_prob - seq_old_log_prob  # (B,)
    
    # 可选：裁剪log_ratio
    if clip_log_ratio > 0:
        log_ratio = torch.clamp(log_ratio, -clip_log_ratio, clip_log_ratio)
    
    # ============================================================
    # Step 2: 自适应温度 & Alpha 策略
    # ============================================================
    adaptive_debug = {}
    use_adaptive_alpha = _cfg("use_adaptive_alpha", False)
    alpha_min = _cfg("alpha_min", 0.35)
    alpha_max = _cfg("alpha_max", 0.9)
    
    if use_adaptive_tau or use_adaptive_alpha:
        with torch.no_grad():
            max_entropy = np.log(vocab_size)
            
            # 使用tensor计算
            h_static_t = torch.clamp(-seq_old_log_prob.detach() / max_entropy, 0.0, 1.0).mean()
            h_dynamic_t = torch.clamp(-seq_log_prob.detach() / max_entropy, 0.0, 1.0).mean()
            confidence_t = 1.0 - h_dynamic_t
            
            # 归一化advantage (作为局部 improvement 信号)
            adv_min_t = advantages.min()
            adv_max_t = advantages.max()
            adv_range = (adv_max_t - adv_min_t).clamp(min=1e-6)
            r_norm_t = ((advantages - adv_min_t) / adv_range).mean()
            reward_signal_t = 2.0 * r_norm_t - 1.0
            p_t = torch.clamp(reward_signal_t, min=0.0)
            
            # 1. 自适应温度
            if use_adaptive_tau:
                base_defense_t = adaptive_tau_alpha * h_static_t
                dynamic_mod_t = adaptive_tau_beta * confidence_t * (-reward_signal_t)
                tau_mod_t = 1.0 + base_defense_t + dynamic_mod_t
                effective_tau = torch.clamp(tau * tau_mod_t, adaptive_tau_min, adaptive_tau_max)
            else:
                effective_tau = tau
                tau_mod_t = torch.tensor(1.0, device=advantages.device)
                
            # 2. 自适应 Alpha (AlphaPO 专属)
            if use_adaptive_alpha:
                # 只有当置信度高且有改进时，才下调 alpha (向 mode-seeking 转换)
                # alpha_t = alpha_max - (alpha_max - alpha_min) * confidence * improvement
                alpha_gate = confidence_t * p_t
                effective_alpha = alpha_max - (alpha_max - alpha_min) * alpha_gate
            else:
                effective_alpha = _cfg("alpha", 0.5)

            # 保存debug信息
            adaptive_debug = {
                "h_static_t": h_static_t, "h_dynamic_t": h_dynamic_t,
                "confidence_t": confidence_t, "reward_signal_t": reward_signal_t,
                "tau_mod_t": tau_mod_t,
                "effective_alpha": effective_alpha if isinstance(effective_alpha, torch.Tensor) else torch.tensor(effective_alpha, device=advantages.device)
            }
    else:
        effective_tau = tau
        effective_alpha = _cfg("alpha", 0.5)
    
    # ============================================================
    # Step 3: 计算anchored scores
    # ============================================================
    # u = log_ratio / τ
    # log_ratio = log(pi) - log(pi_ref)
    # anchored_scores represents the deviation from the reference model
    
    anchored_scores = log_ratio / effective_tau
    
    if clip_anchored_score > 0:
        anchored_scores = torch.clamp(anchored_scores, -clip_anchored_score, clip_anchored_score)
    
    # ============================================================
    # Step 4: Prepare Grouped Tensors (Sort & Reshape)
    # ============================================================
    # 统一处理：无论 Index Mode 还是 Reshape Mode，都整理为 (P, G) 形状
    # 这比 scatter 操作更高效，且支持所有 Loss 变体
    
    batch_size = anchored_scores.shape[0]
    
    # 检查是否有预计算的 q_target
    q_target_precomputed = kwargs.get("q_target", None)
    use_precomputed_q = q_target_precomputed is not None
    
    # 检查是否有完整的组结构（batch_size 是 num_generations 的整数倍）
    has_complete_groups = (batch_size % num_generations == 0) and (batch_size >= num_generations)
    is_flattened = not has_complete_groups
    
    if is_flattened and not use_precomputed_q:
        raise ValueError(f"ADPO requires batch_size divisible by num_generations or pre-computed q_target. Got {batch_size}, {num_generations}")
    
    num_prompts = batch_size // num_generations if has_complete_groups else 0
    
    # ============================================================
    # Unified Grouping: Always use sorted_idx (pre-computed or computed on-demand)
    # This eliminates the redundant Reshape Mode vs Index Mode branches
    # ============================================================
    index = kwargs.get("index", None)
    sorted_idx = kwargs.get("sorted_idx", None)
    
    # Get or compute sorted_idx
    if sorted_idx is None:
        if index is not None:
            index_tensor = as_torch_index(index, device=anchored_scores.device)
            sorted_idx = torch.argsort(index_tensor)
        else:
            # No index provided, assume data is already in order
            sorted_idx = torch.arange(batch_size, device=anchored_scores.device)
    
    # Apply sorting and reshape
    if has_complete_groups:
        # Full group structure: reshape to (P, G)
        u_grouped = anchored_scores[sorted_idx].view(num_prompts, num_generations)
        adv_grouped = advantages[sorted_idx].view(num_prompts, num_generations)
        old_log_prob_grouped = seq_old_log_prob[sorted_idx].view(num_prompts, num_generations)
        q_grouped = q_target_precomputed[sorted_idx].view(num_prompts, num_generations) if use_precomputed_q else None
    else:
        # Partial batch (flattened): keep as (B,)
        u_grouped = anchored_scores[sorted_idx]
        adv_grouped = advantages[sorted_idx]
        old_log_prob_grouped = seq_old_log_prob[sorted_idx]
        q_grouped = q_target_precomputed[sorted_idx] if use_precomputed_q else None
            
    # ============================================================
    # Step 5: Compute q_target (if not provided)
    # ============================================================
    if q_grouped is None:
        # Dense Softmax on (P, G)
        q_grouped = F.softmax(adv_grouped / beta_reward, dim=-1)

    # ============================================================
    # Step 6: Compute Loss (on (P, G) tensors or (B,) tensors)
    # ============================================================
    
    # Track if on-policy gradient fix is used (for AlphaPO)
    alphapo_on_policy_fix_used = False
    
    if loss_variant == "plackett_luce":
        # Plackett-Luce 需要完整的组结构来进行 ranking
        if use_precomputed_q and u_grouped.dim() == 1:
             raise ValueError("Plackett-Luce loss requires full group structure (P, G), incompatible with partial batches.")
        
        # Plackett-Luce (ListMLE)
        loss = _compute_plackett_luce_loss(
            u_grouped, 
            adv_grouped, 
            use_poly_loss=use_poly_loss, 
            epsilon=poly_epsilon,
            top_k=pl_top_k,
            temperature=pl_temperature,
            label_smoothing=pl_label_smoothing,
        )
        
        # Metrics
        term_alignment = torch.tensor(0.0, device=anchored_scores.device)
        term_logsumexp = torch.tensor(0.0, device=anchored_scores.device)
        u_center_val = torch.tensor(0.0, device=anchored_scores.device)
        
    else:
        # Softmax / Scaled / Direct / Decoupled
        
        # 如果是 (B,) 维度的预计算模式，不需要 view 也不需要 mean(dim=-1)
        is_flattened = u_grouped.dim() == 1
        
        # Apply unified formula: u = (log_prob - A·old_log_prob - B·q - C) / tau
        # 通过调整 A, B, C 实现不同变体
        
        # 1. Reconstruct current_log_prob (pi)
        # u_grouped = (pi - pi_ref) / tau  => pi = u_grouped * tau + pi_ref
        current_log_prob = u_grouped * effective_tau + old_log_prob_grouped
        
        # 2. Compute u_for_loss with coefficients
        numerator = current_log_prob - softmax_coef_A * old_log_prob_grouped - softmax_coef_B * q_grouped.detach() - softmax_coef_C
        u_for_loss = numerator / effective_tau

        # Q-Weighted Centering
        if use_q_center:
            # u_center = sum(q * u)
            # 对于 flattened，这需要按组求和。但如果是 partial batch，我们无法正确计算 u_center。
            # 这里我们假设如果用了 use_precomputed_q，则意味着我们无法在当前 batch 计算 u_center。
            # 或者，u_center 应该在预计算阶段计算好传进来？
            # 暂时禁用 flattened 模式下的 u_center，或者如果不影响梯度的话忽略它
            if not is_flattened:
                u_center = (q_grouped.detach() * u_for_loss).sum(dim=-1, keepdim=True).detach()  # (P, 1) - MUST detach!
                u_centered = u_for_loss - u_center
            else:
                # 警告：Partial Batch 无法正确计算 Q-Center，这里跳过 centering
                u_centered = u_for_loss
                u_center = torch.zeros_like(u_for_loss)
        else:
            u_centered = u_for_loss
            u_center = torch.zeros_like(u_for_loss)

        # ============================================================
        # 延迟 Softmax CE 模式：支持跨 micro-batch 的组计算
        # ============================================================
        # 检查是否有预计算的 p_target（通过 kwargs 传入）
        p_target_precomputed = kwargs.get("p_target", None)
        # 兼容旧代码：q_target 和 p_target 混用
        if p_target_precomputed is None and q_target_precomputed is not None:
             p_target_precomputed = q_target_precomputed

        use_delayed_softmax = _cfg("use_delayed_softmax", False)
        
        if loss_variant == "decoupled":
            # =====================================================
            # Decoupled Loss: L = sum(-q*u + 1/G * exp(u))
            # = -q.u + mean(exp(u))
            # 
            # Gradient: -q + 1/G * exp(u)
            # This pushes exp(u)/G towards q
            # =====================================================
            
            # Term 1: -q * u
            if not is_flattened:
                term1 = -(q_grouped.detach() * u_centered).sum(dim=-1)
                term2 = torch.exp(u_centered).mean(dim=-1)
            else:
                # Flattened 模式下，q 和 u 都是 (B,)
                # Loss 是逐样本计算的，最后求 mean
                # 注意：Decoupled 原公式是 sum(-q*u) + mean(exp(u)) 针对一组
                # 对于单样本 i: L_i = -q_i * u_i + 1/G * exp(u_i)
                # 这里的 1/G 系数需要补上
                term1 = -(q_grouped.detach() * u_centered)
                term2 = torch.exp(u_centered) / num_generations # 假设 num_generations 是全局常数
            
            loss_per_prompt = term1 + term2
            loss = loss_per_prompt.mean()
            
            # Metrics
            if not is_flattened:
                term_alignment = (q_grouped.detach() * u_centered).sum(dim=-1).mean().detach()
                term_logsumexp = term2.mean().detach()
                log_Z = torch.logsumexp(u_centered, dim=-1)
            else:
                term_alignment = (q_grouped.detach() * u_centered).mean().detach() * num_generations
                term_logsumexp = term2.mean().detach() * num_generations
                log_Z = torch.tensor(0.0) # 无法计算组内 logsumexp
            
            u_center_val = u_center.mean().detach()

        elif p_target_precomputed is not None and (use_delayed_softmax or use_precomputed_q):
            # =====================================================
            # 预计算 Q 模式
            # 
            # 关键修复：当有完整组结构时，实时计算 p = softmax(u)
            # 而不是使用预计算的 p_target（可能等于 q，导致梯度为 0）
            # =====================================================
            
            if not is_flattened:
                # 有完整的组结构 (P, G)，可以实时计算 p
                # 使用标准 Softmax Loss：L = -q·u + logsumexp(u)
                # 梯度：∂L/∂u = softmax(u) - q = p - q (正确的梯度)
                log_Z = torch.logsumexp(u_centered, dim=-1)  # (P,)
                loss_per_prompt = -(q_grouped.detach() * u_centered).sum(dim=-1) + log_Z
                loss = loss_per_prompt.mean()
                
                if loss_variant == "scaled":
                    loss = loss * grad_scale_factor
                
                # Metrics
                term_alignment = (q_grouped.detach() * u_centered).sum(dim=-1).mean().detach()
                term_logsumexp = torch.exp(log_Z).mean().detach() / num_generations
            else:
                # Flattened 模式 (B,)，无法实时计算组内 softmax
                # 必须使用 DelayedGroupSoftmaxCE（依赖预计算的 p）
                # 注意：这种情况下预计算的 p 必须正确，否则梯度会有问题！
                if index is not None:
                    p_grouped = p_target_precomputed[sorted_idx]
                else:
                    p_grouped = p_target_precomputed
                
                loss = DelayedGroupSoftmaxCE.apply(u_centered, q_grouped.detach(), p_grouped)
                
                if loss_variant == "scaled":
                    loss = loss * grad_scale_factor
                
                # Metrics (limited in flattened mode)
                term_alignment = (q_grouped.detach() * u_centered).mean().detach() * num_generations
                term_logsumexp = torch.tensor(0.0, device=u_centered.device)
                log_Z = torch.tensor(0.0, device=u_centered.device)

            u_center_val = u_center.mean().detach()
            
        elif loss_variant == "alphapo":
            # =====================================================
            # Alpha-Divergence Preference Optimization (AlphaPO)
            # L_alpha = E[ D_alpha(q || p) ]
            # grad = -1/alpha * sum_{i} [ p(i) * (q(i)/p(i))^alpha * grad log p(i) ]
            #      = -1/alpha * sum_{i} [ q(i)^alpha * p(i)^(1-alpha) * grad log p(i) ]
            #
            # Supports both full batch (P, G) and partial batch (B,) modes.
            # Partial batch mode requires pre-computed q_target.
            # =====================================================
            
            # 使用自适应后的有效 alpha
            alpha = effective_alpha
            eps = 1e-10
            
            # Detect on-policy case: log_ratio ≈ 0 OR u_centered ≈ 0
            # This happens when:
            # 1. old_log_prob = log_prob.detach() (on-policy mode)
            # 2. log_ratio is small due to policy being close to anchor
            # In either case, u_centered ≈ 0 causes softmax to become uniform,
            # leading to constant log_p = -log(G) and zero gradients.
            log_ratio_mean = log_ratio.abs().mean()
            u_centered_std = u_centered.std() if has_complete_groups else u_grouped.std()
            
            # Check both conditions: log_ratio and u_centered
            # If u_centered has very low variance, softmax → uniform → zero gradient
            # Thresholds based on observed values:
            # - log_ratio_mean ~ 0.001 from first step logs
            # - anchored_score_std ~ 0.002 from first step logs (without advantage)
            # 
            # FIXED: Change to AND condition. Even if log_ratio is small, if u_centered has 
            # enough variance (e.g. from advantage weighting), we can use standard AlphaPO loss.
            # Only when BOTH are small do we risk zero gradients.
            is_on_policy_case = (log_ratio_mean < 0.01) and (u_centered_std < 0.01)
            
            if is_on_policy_case:
                print(f"[AlphaPO] On-policy fix triggered! log_ratio={log_ratio_mean:.6f}, u_std={u_centered_std:.6f}")
            
            if is_flattened:
                # =====================================================
                # Partial Batch Mode: Support for micro_batch < num_generations
                # 
                # In this mode, we cannot compute softmax(u) because group members
                # are distributed across different micro-batches.
                # 
                # Solution: Use approximation for p in mix_prob calculation.
                # For ppo_epochs=1 (on-policy), at training start:
                #   - log_ratio ≈ 0 (policy close to anchor)
                #   - p = softmax(log_ratio / tau) ≈ Uniform(1/G)
                #
                # So we approximate: mix_prob ≈ q^α * Uniform^(1-α) = q^α * (1/G)^(1-α)
                # This preserves the advantage signal (through q) while being mathematically sound.
                # =====================================================
                
                if q_grouped is None:
                    raise ValueError("AlphaPO partial batch mode requires pre-computed q_target.")
                
                # Check if p_target was pre-computed
                p_target_precomputed = kwargs.get("p_target", None)
                
                if p_target_precomputed is not None:
                    # Use pre-computed p_target
                    p_approx = p_target_precomputed[sorted_idx]
                else:
                    # Approximate p as Uniform (valid when log_ratio ≈ 0)
                    # p ≈ 1/G for each sample in the group
                    p_approx = torch.full_like(q_grouped, 1.0 / num_generations)
                
                # Compute mix_prob = q^alpha * p^(1-alpha)
                # Note: This gives q^α * (1/G)^(1-α), which emphasizes high-advantage samples
                mix_prob = torch.pow(q_grouped.detach() + eps, alpha) * torch.pow(p_approx.detach() + eps, 1.0 - alpha)
                
                # Get seq_log_prob in sorted order
                seq_log_prob_sorted = seq_log_prob[sorted_idx]
                
                # AlphaPO-style loss: weighted sum of log_probs
                # L = -1/α * Σ [mix_prob * seq_log_prob]
                # Gradient flows through seq_log_prob -> log_prob -> model parameters
                loss = -(1.0 / alpha) * (mix_prob.detach() * seq_log_prob_sorted).mean()
                
                # Metrics (limited in flattened mode)
                term_alignment = (q_grouped.detach() * u_grouped).mean().detach()
                term_logsumexp = torch.tensor(0.0, device=u_grouped.device)
                log_Z = torch.tensor(0.0, device=u_grouped.device)
                u_center_val = torch.tensor(0.0, device=u_grouped.device)
                
            else:
                # =====================================================
                # Full Batch Mode: Standard AlphaPO with (P, G) tensors
                # =====================================================
                
                # Compute p_theta (current policy distribution)
                # Use F.log_softmax for better performance (fused kernel, single pass)
                log_p = F.log_softmax(u_centered, dim=-1)  # (P, G)
                p_theta = torch.exp(log_p)
                log_Z = u_centered.logsumexp(dim=-1, keepdim=True)  # Only for metrics
                
                # Compute importance weights (detached)
                # weight = q^alpha * p^(1-alpha)
                mix_prob = torch.pow(q_grouped.detach() + eps, alpha) * torch.pow(p_theta.detach() + eps, 1.0 - alpha)
                
                if is_on_policy_case:
                    # On-policy fix: Use REINFORCE-style gradient estimation
                    # When log_ratio ≈ 0, standard AlphaPO has zero gradient
                    alphapo_on_policy_fix_used = True
                    
                    # Get seq_log_prob in grouped form (P, G)
                    seq_log_prob_grouped = seq_log_prob[sorted_idx].view(num_prompts, num_generations)
                    
                    # Normalize mix_prob for stability
                    mix_prob_normalized = mix_prob / (mix_prob.sum(dim=-1, keepdim=True) + eps)
                    loss_per_prompt = -(mix_prob_normalized.detach() * seq_log_prob_grouped).sum(dim=-1)
                    loss = loss_per_prompt.mean()
                    
                    # Entropy bonus to encourage exploration
                    entropy_bonus = -(p_theta * log_p).sum(dim=-1).mean()
                    loss = loss - 0.01 * entropy_bonus
                else:
                    # Standard AlphaPO loss (off-policy case)
                    # L = - 1/alpha * sum( mix_prob * log_p )
                    loss_per_prompt = -(1.0 / alpha) * (mix_prob * log_p).sum(dim=-1)
                    loss = loss_per_prompt.mean()
                    
                    # DEBUG: Aggressive Gradient Check
                    print(f"\n[AlphaPO_DEBUG_STEP] loss value: {loss.item()}")
                    print(f"[AlphaPO_DEBUG_STEP] loss.requires_grad: {loss.requires_grad}")
                    print(f"[AlphaPO_DEBUG_STEP] loss.grad_fn: {loss.grad_fn}")
                    
                    if not loss.requires_grad:
                        print("!!!!!! LOSS HAS NO GRADIENT - BROKEN CHAIN !!!!!!")
                        print(f"  log_p.requires_grad={log_p.requires_grad} grad_fn={log_p.grad_fn}")
                        print(f"  mix_prob.requires_grad={mix_prob.requires_grad}")
                        print(f"  u_centered.requires_grad={u_centered.requires_grad} grad_fn={u_centered.grad_fn}")
                        print(f"  u_grouped.requires_grad={u_grouped.requires_grad} grad_fn={u_grouped.grad_fn}")
                        print(f"  anchored_scores.requires_grad={anchored_scores.requires_grad} grad_fn={anchored_scores.grad_fn}")
                        print(f"  log_ratio.requires_grad={log_ratio.requires_grad} grad_fn={log_ratio.grad_fn}")
                        print(f"  log_prob input requires_grad={log_prob.requires_grad} grad_fn={log_prob.grad_fn}")
                        # Raise error to stop training and force user to see this
                        # raise RuntimeError("AlphaPO Loss has no gradient! Check logs above.")

                    # Also check if gradient is zero (vanishing)
                    # We can't check .grad yet, but we can check values
                    if u_centered.std() < 1e-6:
                         print(f"!!!!!! u_centered VANISHED: std={u_centered.std().item()} !!!!!!")

                
                # Metrics
                term_alignment = (q_grouped.detach() * u_centered).sum(dim=-1).mean().detach()
                term_logsumexp = torch.exp(log_Z).mean().detach() / num_generations
                u_center_val = u_center.mean().detach()

        else:
            # =====================================================
            # 标准模式：实时计算 logsumexp（需要组内完整）
            # =====================================================
            if is_flattened:
                 raise ValueError("Standard Softmax loss requires full group structure (P, G). Use precomputed q/p for partial batches.")

            log_Z = torch.logsumexp(u_centered, dim=-1)  # (P,)
            loss_per_prompt = -(q_grouped.detach() * u_centered).sum(dim=-1) + log_Z
            loss = loss_per_prompt.mean()
            
            if loss_variant == "scaled":
                loss = loss * grad_scale_factor

            # Metrics
            term_alignment = (q_grouped.detach() * u_centered).sum(dim=-1).mean().detach()
            term_logsumexp = torch.exp(log_Z).mean().detach() / num_generations  # approx Z/G
            u_center_val = u_center.mean().detach()
    # ============================================================
    # Step 7: 梯度裁剪 & Logit Reg
    # ============================================================
    if grad_clip_value > 0:
        with torch.no_grad():
            scale = torch.clamp(loss.abs() / grad_clip_value, min=1.0)
        loss = loss / scale
    
    if logit_reg_coef > 0:
        # L_z = coef * (log Z)^2
        # We re-compute log_Z mean if not available (e.g. PL mode doesn't compute it by default)
        if loss_variant == "plackett_luce":
             # For PL, u_grouped is not centered, so logsumexp might be large
             # We compute it just for regularization
             lz = torch.logsumexp(u_grouped, dim=-1).mean()
             logit_reg_loss = logit_reg_coef * (lz ** 2)
        else:
             # For Softmax, we have log_Z (per prompt)
             logit_reg_loss = logit_reg_coef * (log_Z.mean() ** 2)
             
        loss = loss + logit_reg_loss
    else:
        logit_reg_loss = torch.tensor(0.0, device=loss.device)
    
    # ============================================================
    # Metrics
    # ============================================================
    with torch.no_grad():
        kl_val = log_ratio.abs().mean().item()
        
        metrics = {
            "actor/ppo_kl": kl_val,
            "actor/pg_clipfrac": 0.0,
            "adpo/loss": loss.detach().item(),
            "adpo/advantage_mean": advantages.mean().item(),
            "adpo/advantage_std": advantages.std().item(),
            "adpo/anchored_score_mean": anchored_scores.mean().item(),
            "adpo/anchored_score_std": anchored_scores.std().item(),
            "adpo/log_ratio_mean": log_ratio.abs().mean().item(),  # Monitor log_ratio component
            "adpo/adv_score_weight": adv_score_weight,  # Current weight setting
            "adpo/effective_tau": float(effective_tau.item() if isinstance(effective_tau, torch.Tensor) else effective_tau),
            "adpo/effective_alpha": float(effective_alpha.item() if isinstance(effective_alpha, torch.Tensor) else effective_alpha),
            "adpo/term_alignment": float(term_alignment.item() if isinstance(term_alignment, torch.Tensor) else term_alignment),
            "adpo/term_logsumexp": float(term_logsumexp.item() if isinstance(term_logsumexp, torch.Tensor) else term_logsumexp),
            "adpo/logit_reg_loss": float(logit_reg_loss.item() if isinstance(logit_reg_loss, torch.Tensor) else logit_reg_loss),
            "adpo/u_center": float(u_center_val.item() if isinstance(u_center_val, torch.Tensor) else u_center_val),
            "adpo/alphapo_on_policy_fix": 1.0 if alphapo_on_policy_fix_used else 0.0,
        }
        
        if adaptive_debug:
            metrics["adpo/h_static"] = adaptive_debug["h_static_t"].item()
            metrics["adpo/h_dynamic"] = adaptive_debug["h_dynamic_t"].item()
            metrics["adpo/confidence"] = adaptive_debug["confidence_t"].item()
            metrics["adpo/reward_signal"] = adaptive_debug["reward_signal_t"].item()
            metrics["adpo/tau_modulation"] = adaptive_debug["tau_mod_t"].item()
            if "effective_alpha" in adaptive_debug:
                metrics["adpo/alpha"] = adaptive_debug["effective_alpha"].item()
    
    return loss, metrics
