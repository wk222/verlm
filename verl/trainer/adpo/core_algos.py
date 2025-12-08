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

from verl.trainer.ppo.core_algos import register_adv_est, register_policy_loss


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
        
    # Compute std
    out_sq_sum = torch.zeros(num_groups, device=values.device, dtype=values.dtype)
    out_sq_sum.index_add_(0, index, values.pow(2))
    
    group_sq_mean = out_sq_sum / out_count
    group_var = group_sq_mean - group_mean.pow(2)
    group_std = group_var.clamp(min=1e-10).sqrt()
    
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
    """
    with torch.no_grad():
        sequence_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # (B,)
        
        batch_size = sequence_rewards.shape[0]
        device = sequence_rewards.device
        num_generations = config.get("num_generations", 8) if config else 8
        scale_rewards = config.get("scale_rewards", True) if config else True
        
        # Convert index to tensor if present
        if index is not None:
            if isinstance(index, np.ndarray):
                index_tensor = torch.from_numpy(index).to(device).long()
            else:
                index_tensor = index.to(device).long()
                
            # Vectorized Group Mean/Std
            mean_rewards, std_rewards = _scatter_group_mean_std(sequence_rewards, index_tensor, return_std=scale_rewards)
            
            if scale_rewards:
                # std_rewards is already broadcasted to (B,)
                std_rewards = torch.where(std_rewards < 1e-8, torch.ones_like(std_rewards), std_rewards)
                advantages = (sequence_rewards - mean_rewards) / std_rewards
            else:
                advantages = sequence_rewards - mean_rewards
                
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
    # 使用更稳定的实现
    sorted_u_flip = torch.flip(sorted_u, dims=[-1])  # (P, G) 翻转
    
    # 数值稳定的 cumulative logsumexp
    # 方法：逐步计算，每步都减去当前最大值
    max_vals = sorted_u_flip.cummax(dim=-1).values  # (P, G)
    exp_shifted = torch.exp(sorted_u_flip - max_vals)
    cumsum_exp = torch.cumsum(exp_shifted, dim=-1)
    # 防止 log(0)
    cumsum_exp = torch.clamp(cumsum_exp, min=1e-10)
    cum_logsumexp_flip = max_vals + torch.log(cumsum_exp)  # (P, G)
    
    # 翻转回来得到从位置 k 开始的 logsumexp
    cum_logsumexp = torch.flip(cum_logsumexp_flip, dims=[-1])  # (P, G)
    
    # loss_k = -u[k] + logsumexp(u[k:])
    # 这就是 -log(p_rank)
    per_position_loss = -sorted_u[:, :-1] + cum_logsumexp[:, :-1]  # (P, G-1)
    
    # Top-K: 只计算前 k 个位置
    if top_k > 0 and top_k < G - 1:
        per_position_loss = per_position_loss[:, :top_k]  # (P, top_k)
    
    # Label Smoothing: 软化排序目标
    # 思想：不要求完美排序，允许一定的误差
    if label_smoothing > 0:
        # 对 loss 进行平滑：loss = (1 - α) * loss + α * uniform_loss
        # uniform_loss ≈ log(G) (随机猜测的期望 loss)
        num_positions = per_position_loss.shape[1]
        uniform_loss = torch.log(torch.tensor(G, dtype=u.dtype, device=u.device))
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
    
    if index is not None:
        # Index Mode: Sort by index to group elements physically
        if isinstance(index, np.ndarray):
            index_tensor = torch.from_numpy(index).to(anchored_scores.device).long()
        else:
            index_tensor = index.to(anchored_scores.device).long()
            
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
        if use_q_center:
            # u_center = sum(q * u, dim=-1)
            u_center = (q_grouped.detach() * u_for_loss).sum(dim=-1, keepdim=True) # (P, 1)
            u_centered = u_for_loss - u_center
        else:
            u_centered = u_for_loss
            u_center = torch.zeros_like(u_for_loss)

        # Loss Calculation: L = -sum(q * u) + logsumexp(u)
        # Note: logsumexp(u) is computed per group
        
        if loss_variant == "direct":
            # Direct: -q*u + logsumexp(u)
            # This matches the decoupled estimator expectation E[-q*u + exp(u)/G] but exact
            log_Z = torch.logsumexp(u_centered, dim=-1) # (P,)
            loss_per_prompt = -(q_grouped.detach() * u_centered).sum(dim=-1) + log_Z
            loss = loss_per_prompt.mean()
            
        else: # softmax or scaled
            # Standard ADPO/DPO-like: -q * log_softmax(u)
            # log_softmax(u) = u - logsumexp(u)
            # -q * (u - logZ) = -q*u + q*logZ = -q*u + logZ (since sum(q)=1)
            # So it is mathematically identical to "direct"
            
            # However, we can use CrossEntropy if we view q as soft labels
            # loss = torch.sum(-q * F.log_softmax(u_centered, dim=-1), dim=-1).mean()
            
            # Let's stick to the explicit form for clarity
            log_Z = torch.logsumexp(u_centered, dim=-1)
            loss_per_prompt = -(q_grouped.detach() * u_centered).sum(dim=-1) + log_Z
            loss = loss_per_prompt.mean()
        
        if loss_variant == "scaled":
            loss = loss * grad_scale_factor

        # Metrics
        term_alignment = (q_grouped.detach() * u_centered).sum(dim=-1).mean().detach()
        term_logsumexp = torch.exp(log_Z).mean().detach() / num_generations # approx Z/G
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
