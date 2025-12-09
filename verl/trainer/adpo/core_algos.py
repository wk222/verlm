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
            if use_delayed_softmax:
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
                    extra_returns["p_target"] = p_target
                
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
    # 系数 A 控制 reference model 的锚定强度 (默认 1.0)
    # 系数 B 控制 target distribution q 的减法项 (默认 0.5)
    # 默认公式: u = log_prob - 1.0 * old_log_prob - 0.5 * q
    softmax_coef_A = _cfg("softmax_coef_A", 1.0)
    softmax_coef_B = _cfg("softmax_coef_B", 0.5)
    
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
    # Step 2: 自适应温度策略（纯tensor操作，避免.item()同步）
    # ============================================================
    # 注意：为避免分布式训练中的同步问题，所有计算保持在tensor上
    # 只在最后输出metrics时才转换为Python值
    adaptive_debug = None
    if use_adaptive_tau:
        with torch.no_grad():
            max_entropy = np.log(vocab_size)
            
            # 使用tensor计算，不调用.item()
            h_static_t = torch.clamp(-seq_old_log_prob.detach() / max_entropy, 0.0, 1.0).mean()
            h_dynamic_t = torch.clamp(-seq_log_prob.detach() / max_entropy, 0.0, 1.0).mean()
            confidence_t = 1.0 - h_dynamic_t
            
            # 归一化advantage
            adv_min_t = advantages.min()
            adv_max_t = advantages.max()
            adv_range = (adv_max_t - adv_min_t).clamp(min=1e-6)
            r_norm_t = ((advantages - adv_min_t) / adv_range).mean()
            reward_signal_t = 2.0 * r_norm_t - 1.0
            
            # 计算温度调节（tensor操作）
            base_defense_t = adaptive_tau_alpha * h_static_t
            dynamic_mod_t = adaptive_tau_beta * confidence_t * (-reward_signal_t)
            tau_mod_t = 1.0 + base_defense_t + dynamic_mod_t
            effective_tau_t = tau * tau_mod_t
            effective_tau_t = torch.clamp(effective_tau_t, adaptive_tau_min, adaptive_tau_max)
            
            # 优化：保持为 Tensor 参与后续计算，避免 .item() 带来的 CPU-GPU 同步阻塞
            effective_tau = effective_tau_t
            
            # 保存debug信息（保持 Tensor 形式）
            adaptive_debug = {
                "h_static_t": h_static_t, "h_dynamic_t": h_dynamic_t,
                "confidence_t": confidence_t, "reward_signal_t": reward_signal_t,
                "tau_mod_t": tau_mod_t
            }
    else:
        effective_tau = tau  # 保持为 Python float
    
    # ============================================================
    # Step 3: 计算anchored scores
    # ============================================================
    anchored_scores = log_ratio / effective_tau  # (B,) 全程 GPU 计算
    
    if clip_anchored_score > 0:
        anchored_scores = torch.clamp(anchored_scores, -clip_anchored_score, clip_anchored_score)
    
    # ============================================================
    # Step 4: Prepare Grouped Tensors (Sort & Reshape)
    # ============================================================
    # 统一处理：无论 Index Mode 还是 Reshape Mode，都整理为 (P, G) 形状
    # 这比 scatter 操作更高效，且支持所有 Loss 变体
    
    batch_size = anchored_scores.shape[0]
    if batch_size % num_generations != 0:
        raise ValueError(f"ADPO requires batch_size divisible by num_generations. Got {batch_size}, {num_generations}")
    
    num_prompts = batch_size // num_generations
    index = kwargs.get("index", None)
    sorted_idx = None  # Not sure Initialize to avoid potential undefined variable issues
    
    if index is not None:
        # Index Mode: Sort by index to group elements physically
        # Use as_torch_index to handle various index types (int, str/uuid, etc.)
        index_tensor = as_torch_index(index, device=anchored_scores.device)
            
        sorted_idx = torch.argsort(index_tensor)
        
        # Reorder inputs to be contiguous groups
        u_grouped = anchored_scores[sorted_idx].view(num_prompts, num_generations)
        adv_grouped = advantages[sorted_idx].view(num_prompts, num_generations)
        
        # 如果有 precomputed q，也需要排序
        q_target_precomputed = kwargs.get("q_target", None)
        if q_target_precomputed is not None:
            q_grouped = q_target_precomputed[sorted_idx].view(num_prompts, num_generations)
        else:
            q_grouped = None
            
        # 用于后续计算的 old_log_prob (如果需要)
        old_log_prob_grouped = seq_old_log_prob[sorted_idx].view(num_prompts, num_generations)
        
    else:
        # Reshape Mode: Assume already grouped
        u_grouped = anchored_scores.view(num_prompts, num_generations)
        adv_grouped = advantages.view(num_prompts, num_generations)
        
        q_target_precomputed = kwargs.get("q_target", None)
        if q_target_precomputed is not None:
            q_grouped = q_target_precomputed.view(num_prompts, num_generations)
        else:
            q_grouped = None
            
        old_log_prob_grouped = seq_old_log_prob.view(num_prompts, num_generations)

    # ============================================================
    # Step 5: Compute q_target (if not provided)
    # ============================================================
    if q_grouped is None:
        # Dense Softmax on (P, G)
        q_grouped = F.softmax(adv_grouped / beta_reward, dim=-1)

    # ============================================================
    # Step 6: Compute Loss (on (P, G) tensors)
    # ============================================================
    
    if loss_variant == "plackett_luce":
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
        # Softmax / Scaled / Direct
        # Apply new formula adjustment: u_new = u - (A-1)·ℓ_ref/τ - B·q/τ
        if softmax_coef_A != 1.0 or softmax_coef_B != 0.0:
            u_for_loss = u_grouped - (softmax_coef_A - 1.0) * old_log_prob_grouped / effective_tau - softmax_coef_B * q_grouped.detach() / effective_tau
        else:
            u_for_loss = u_grouped

        # Q-Weighted Centering
        # 注意：u_center 必须 detach！否则当 softmax(u_centered) ≈ q 时梯度消失
        # 数学推导：
        #   L = -(q·u_centered).sum() + logsumexp(u_centered)
        #   如果 u_center 不 detach，u_centered = u - u_center，而 u_center = (q·u).sum()
        #   展开后 -(q·u).sum() + (q·u).sum() = 0，alignment term 被抵消！
        #   梯度变成 softmax(u_centered)·(I - q^T)，当 softmax ≈ q 时为 0
        if use_q_center:
            # u_center = sum(q * u, dim=-1)
            u_center = (q_grouped.detach() * u_for_loss).sum(dim=-1, keepdim=True).detach()  # (P, 1) - MUST detach!
            u_centered = u_for_loss - u_center
        else:
            u_centered = u_for_loss
            u_center = torch.zeros_like(u_for_loss)

        # ============================================================
        # 延迟 Softmax CE 模式：支持跨 micro-batch 的组计算
        # ============================================================
        # 检查是否有预计算的 p_target（通过 kwargs 传入）
        p_target_precomputed = kwargs.get("p_target", None)
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
            term1 = -(q_grouped.detach() * u_centered).sum(dim=-1)
            
            # Term 2: mean(exp(u))
            term2 = torch.exp(u_centered).mean(dim=-1)
            
            loss_per_prompt = term1 + term2
            loss = loss_per_prompt.mean()
            
            # Metrics
            term_alignment = (q_grouped.detach() * u_centered).sum(dim=-1).mean().detach()
            term_logsumexp = term2.mean().detach()
            u_center_val = u_center.mean().detach()
            
            # For logging/reg purposes, compute logsumexp
            log_Z = torch.logsumexp(u_centered, dim=-1)

        elif p_target_precomputed is not None and use_delayed_softmax:
            # =====================================================
            # 延迟模式：使用预计算的 p_target
            # 
            # 优势：
            # 1. 不需要 batch_size >= num_generations
            # 2. 可以在 advantage 计算阶段（所有组成员已收集）预计算 p
            # 3. 梯度仍然是正确的 (p - q)
            # 
            # 使用方法：
            # 1. 在 compute_adpo_advantages 时同时计算 p_target
            # 2. 将 p_target 存入 DataProto 并传递给 actor
            # 3. actor 调用 loss 时传入 p_target=...
            # =====================================================
            
            # 重新排序 p_target 以匹配 u_centered 的顺序
            if index is not None:
                p_grouped = p_target_precomputed[sorted_idx].view(num_prompts, num_generations)
            else:
                p_grouped = p_target_precomputed.view(num_prompts, num_generations)
            
            # 使用自定义 autograd function
            loss = DelayedGroupSoftmaxCE.apply(u_centered, q_grouped.detach(), p_grouped)
            
            if loss_variant == "scaled":
                loss = loss * grad_scale_factor
            
            # Metrics（使用预计算的 p）
            term_alignment = (q_grouped.detach() * u_centered).sum(dim=-1).mean().detach()
            # p_grouped 已经是 softmax，所以 log_Z ≈ log(sum(exp(u)))
            with torch.no_grad():
                log_Z_approx = torch.logsumexp(u_centered.detach(), dim=-1)
                term_logsumexp = torch.exp(log_Z_approx).mean() / num_generations
            u_center_val = u_center.mean().detach()
            log_Z = log_Z_approx  # 用于后续的正则化
            
        else:
            # =====================================================
            # 标准模式：实时计算 logsumexp（需要组内完整）
            # 
            # Loss Calculation: L = -sum(q * u) + logsumexp(u)
            # 梯度: ∂L/∂u = softmax(u) - q = p - q
            # =====================================================
            
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
            "adpo/effective_tau": float(effective_tau.item() if isinstance(effective_tau, torch.Tensor) else effective_tau),
            "adpo/term_alignment": float(term_alignment.item() if isinstance(term_alignment, torch.Tensor) else term_alignment),
            "adpo/term_logsumexp": float(term_logsumexp.item() if isinstance(term_logsumexp, torch.Tensor) else term_logsumexp),
            "adpo/logit_reg_loss": float(logit_reg_loss.item() if isinstance(logit_reg_loss, torch.Tensor) else logit_reg_loss),
            "adpo/u_center": float(u_center_val.item() if isinstance(u_center_val, torch.Tensor) else u_center_val),
        }
        
        if adaptive_debug:
            metrics["adpo/h_static"] = adaptive_debug["h_static_t"].item()
            metrics["adpo/h_dynamic"] = adaptive_debug["h_dynamic_t"].item()
            metrics["adpo/confidence"] = adaptive_debug["confidence_t"].item()
            metrics["adpo/reward_signal"] = adaptive_debug["reward_signal_t"].item()
            metrics["adpo/tau_modulation"] = adaptive_debug["tau_mod_t"].item()
    
    return loss, metrics
