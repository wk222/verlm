# Copyright 2025 ADPO Algorithm Author
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ADPO (Anchored Direct Preference Optimization) - 高效向量化实现
"""
ADPO核心算法 - 保留完整功能的向量化实现

功能：
- 多种loss变体（pairwise, direct, infonce, plackett_luce, softmax）
- 自适应温度策略
- 梯度裁剪
- 完全向量化（无Python循环）
"""

from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from verl.trainer.ppo.core_algos import register_adv_est, register_policy_loss


@register_adv_est("adpo")
def compute_adpo_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: DictConfig,
    index: Optional[np.ndarray] = None,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算ADPO advantages（序列级别，返回 (B,) 维度）
    """
    with torch.no_grad():
        sequence_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # (B,)
        
        batch_size = sequence_rewards.shape[0]
        device = sequence_rewards.device
        num_generations = config.get("num_generations", 8) if config else 8
        
        # 高效路径：reshape计算组内均值
        # 注意：batch_size % num_generations 在所有rank上应该一致（相同配置）
        # 但为了安全，我们统一处理
        num_prompts = batch_size // num_generations
        if num_prompts > 0 and batch_size == num_prompts * num_generations:
            rewards_reshaped = sequence_rewards.view(num_prompts, num_generations)
            mean_rewards = rewards_reshaped.mean(dim=-1, keepdim=True)
            
            # 强制开启标准化（Z-Score），这对 Direct/Softmax 至关重要
            # 即使 config 里没写，我们也默认开启，除非显式设为 False
            scale_rewards = config.get("scale_rewards", True) if config else True
            
            if scale_rewards:
                std_rewards = rewards_reshaped.std(dim=-1, keepdim=True)
                # 避免除以0：如果 std < 1e-8，说明所有 reward 都一样，此时 advantage 设为 0
                std_rewards = torch.where(std_rewards < 1e-8, torch.ones_like(std_rewards), std_rewards)
                advantages_reshaped = (rewards_reshaped - mean_rewards) / std_rewards
            else:
                advantages_reshaped = rewards_reshaped - mean_rewards
            
            advantages = advantages_reshaped.view(-1)
        else:
            # Fallback（理论上不应该触发，除非配置错误）
            advantages = sequence_rewards - sequence_rewards.mean()
        
    return advantages, sequence_rewards


def _compute_plackett_luce_loss(u: torch.Tensor, adv: torch.Tensor, use_poly_loss: bool = False, epsilon: float = 1.0) -> torch.Tensor:
    """
    Plackett-Luce (ListMLE) loss - 完全向量化实现（无循环）
    
    Args:
        u: (P, G) anchored scores
        adv: (P, G) advantages
        use_poly_loss: 是否开启 Poly-Loss 修正
        epsilon: Poly-Loss 的权重系数
        
    Returns:
        loss: scalar
        
    数学形式:
        Standard: loss = -sum_k [u[σ(k)] - logsumexp(u[σ(k:)])]
        Poly: loss = Standard + epsilon * (1 - p_rank)
        其中 p_rank = exp(-Standard_k)
        
    向量化技巧：
        使用 cumulative logsumexp 从后往前计算
    """
    P, G = u.shape
    
    # 按advantage排序
    sorted_indices = torch.argsort(adv, dim=-1, descending=True)  # (P, G)
    sorted_u = torch.gather(u, 1, sorted_indices)  # (P, G)
    
    # 完全向量化的cumulative logsumexp（从后往前）
    # 翻转 -> cumulative操作 -> 翻转回来
    sorted_u_flip = torch.flip(sorted_u, dims=[-1])  # (P, G) 翻转
    
    # 计算cumulative logsumexp: log(cumsum(exp(x)))
    # 使用数值稳定的方法
    max_vals = sorted_u_flip.cummax(dim=-1).values  # (P, G)
    exp_shifted = torch.exp(sorted_u_flip - max_vals)
    cumsum_exp = torch.cumsum(exp_shifted, dim=-1)
    cum_logsumexp_flip = max_vals + torch.log(cumsum_exp)  # (P, G)
    
    # 翻转回来得到从位置k开始的logsumexp
    cum_logsumexp = torch.flip(cum_logsumexp_flip, dims=[-1])  # (P, G)
    
    # loss_k = -u[k] + logsumexp(u[k:])，只计算前G-1个位置
    # 这就是 -log(p_rank)
    per_position_loss = -sorted_u[:, :-1] + cum_logsumexp[:, :-1]  # (P, G-1)
    
    if use_poly_loss:
        # Poly-Ranking Loss: L = -log(p) + epsilon * (1 - p)
        # per_position_loss is -log(p)
        p_rank = torch.exp(-per_position_loss)
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
    - "pairwise": DPO风格 -log σ(u_w - u_l) ⭐推荐
    - "plackett_luce": P-L模型/ListMLE，完整排序对齐
    - "direct": -q·u + logsumexp(u)
    - "infonce": -u_best + logsumexp(u)
    - "softmax": 原始ADPO
    - "scaled": 原始ADPO × scale
    
    功能:
    - 自适应温度策略
    - 梯度裁剪
    - 长度归一化
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
    loss_variant = _cfg("loss_variant", "pairwise")
    clip_anchored_score = _cfg("clip_anchored_score", 10.0)
    use_length_normalization = _cfg("use_length_normalization", True)
    grad_scale_factor = _cfg("grad_scale_factor", 20.0)
    plackett_luce_top_k = _cfg("plackett_luce_top_k", 3)
    
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

    # Poly-Loss Config
    # 默认开启 Poly-Loss (特别是对于 P-L Loss)
    use_poly_loss = _cfg("use_poly_loss", True)
    poly_epsilon = _cfg("poly_epsilon", 1.0)
    
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
    # Step 4: 分组处理 - 支持 micro_batch_size < num_generations
    # ============================================================
    batch_size = anchored_scores.shape[0]
    
    # 获取预计算的 q_target 和 index（从 kwargs，由 trainer 传入）
    q_target_precomputed = kwargs.get("q_target", None)
    index = kwargs.get("index", None)
    
    # 判断是否可以使用高效的 reshape 模式
    can_use_reshape = (batch_size % num_generations == 0) and (batch_size >= num_generations)
    
    if can_use_reshape:
        # 高效路径：直接 reshape
        num_prompts = batch_size // num_generations
        G = num_generations
        
        u = anchored_scores.view(num_prompts, G)  # (P, G)
        adv = advantages.view(num_prompts, G)     # (P, G)
        q = F.softmax(adv / beta_reward, dim=-1)  # (P, G)
        
        use_index_mode = False
    else:
        # Index-based 模式：支持 micro_batch < num_generations
        # 使用预计算的 q_target（如果有）
        if q_target_precomputed is not None:
            q_flat = q_target_precomputed
        else:
            # Fallback: 计算 softmax（这种情况下可能不完全正确，但至少有梯度）
            q_flat = F.softmax(advantages / beta_reward, dim=-1)
        
        u_flat = anchored_scores
        adv_flat = advantages
        
        use_index_mode = True
        G = num_generations  # 仍然需要这个参数
    
    # ============================================================
    # Step 5: 计算loss
    # ============================================================
    # 用于详细 logging
    term_alignment = 0.0
    term_logsumexp = 0.0
    u_center_val = 0.0
    
    if not use_index_mode:
        # ============================================================
        # Reshape 模式：(P, G) 张量，高效向量化
        # ============================================================
        if loss_variant == "pairwise":
            u_diff = u.unsqueeze(-1) - u.unsqueeze(-2)  # (P, G, G)
            adv_diff = adv.unsqueeze(-1) - adv.unsqueeze(-2)  # (P, G, G)
            pair_mask = (adv_diff > 0).float()
            pair_loss = F.softplus(-u_diff) * pair_mask
            num_pairs = pair_mask.sum().clamp(min=1)
            loss = pair_loss.sum() / num_pairs
            
        elif loss_variant == "plackett_luce":
            loss = _compute_plackett_luce_loss(u, adv, use_poly_loss=use_poly_loss, epsilon=poly_epsilon)
            
        elif loss_variant == "direct":
            # Q-Weighted Centering: 消除平移不变性
            if use_q_center:
                u_center = (q.detach() * u).sum(dim=-1, keepdim=True)  # (P, 1)
                u_centered = u - u_center  # (P, G)
            else:
                u_centered = u
                u_center = torch.zeros(1, device=u.device)
            
            # 计算 Direct Loss (使用 centered logits)
            weighted_u = (q.detach() * u_centered).sum(dim=-1)
            logsumexp_u = torch.logsumexp(u_centered, dim=-1)
            loss = (-weighted_u + logsumexp_u).mean()
            
            # 记录分项
            term_alignment = weighted_u.mean().detach()
            term_logsumexp = logsumexp_u.mean().detach()
            u_center_val = u_center.mean().detach()
            
        elif loss_variant == "infonce":
            best_idx = adv.argmax(dim=-1)
            u_best = u.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
            logsumexp_u = torch.logsumexp(u, dim=-1)
            loss = (-u_best + logsumexp_u).mean()
            
        elif loss_variant == "softmax":
            log_p = F.log_softmax(u, dim=-1)
            if use_poly_loss:
                p_tilde = torch.exp(log_p)
                # Poly-ADPO: sum(q * (-log_p + epsilon * (1 - p)))
                ce_term = -(q.detach() * log_p).sum(dim=-1)
                poly_term = (q.detach() * poly_epsilon * (1.0 - p_tilde)).sum(dim=-1)
                loss = (ce_term + poly_term).mean()
            else:
                loss = -(q.detach() * log_p).sum(dim=-1).mean()
            
        elif loss_variant == "scaled":
            log_p = F.log_softmax(u, dim=-1)
            if use_poly_loss:
                p_tilde = torch.exp(log_p)
                ce_term = -(q.detach() * log_p).sum(dim=-1)
                poly_term = (q.detach() * poly_epsilon * (1.0 - p_tilde)).sum(dim=-1)
                loss = (ce_term + poly_term).mean() * grad_scale_factor
            else:
                loss = -(q.detach() * log_p).sum(dim=-1).mean() * grad_scale_factor
            
        else:
            raise ValueError(f"Unknown loss_variant: {loss_variant}")
        
        # 用于 metrics
        adv_for_metrics = adv
        u_for_metrics = u
        q_for_metrics = q
        
    else:
        # ============================================================
        # Index 模式：支持 micro_batch < num_generations
        # 使用 softmax 变体（原始 ADPO），因为它最稳定
        # ============================================================
        # 注意：在此模式下，我们使用预计算的 q_target
        # 计算 log_softmax(u) 需要按组进行
        
        # ============================================================
        # Q-Weighted Centering (Index Mode)
        # ============================================================
        u_flat_centered = u_flat.clone()
        if use_q_center and index is not None:
            from collections import defaultdict
            id2indices = defaultdict(list)
            for i in range(batch_size):
                id2indices[index[i]].append(i)
            
            # 对每组应用 Q-weighted centering
            for idx, indices in id2indices.items():
                indices_tensor = torch.tensor(indices, device=u_flat.device, dtype=torch.long)
                group_u = u_flat[indices_tensor]
                group_q = q_flat[indices_tensor]
                # Q-weighted mean
                group_u_center = (group_q.detach() * group_u).sum()
                # Center
                u_flat_centered[indices_tensor] = group_u - group_u_center
        
        # ============================================================
        # Compute Loss (Index Mode)
        # ============================================================
        if index is not None:
            # 使用 index 分组计算 log_softmax
            from collections import defaultdict
            id2indices = defaultdict(list)
            for i in range(batch_size):
                id2indices[index[i]].append(i)
            
            # 计算每组的 log_softmax (使用 centered logits)
            log_p_flat = torch.zeros_like(u_flat_centered)
            logsumexp_flat = torch.zeros_like(u_flat_centered)
            for idx, indices in id2indices.items():
                indices_tensor = torch.tensor(indices, device=u_flat_centered.device, dtype=torch.long)
                group_u = u_flat_centered[indices_tensor]
                group_log_p = F.log_softmax(group_u, dim=0)
                log_p_flat[indices_tensor] = group_log_p
                # 记录 logsumexp (用于 Z-Loss)
                logsumexp_flat[indices_tensor] = torch.logsumexp(group_u, dim=0)
            
            # 交叉熵 loss
            if use_poly_loss:
                p_flat = torch.exp(log_p_flat)
                # Poly-ADPO: sum(q * (-log_p + epsilon * (1 - p)))
                ce_term = -(q_flat.detach() * log_p_flat)
                poly_term = q_flat.detach() * poly_epsilon * (1.0 - p_flat)
                loss = (ce_term + poly_term).sum() / batch_size
            else:
                loss = -(q_flat.detach() * log_p_flat).sum() / batch_size
        else:
            # 没有 index，fallback 到整体 softmax（可能不正确但有梯度）
            log_p_flat = F.log_softmax(u_flat_centered, dim=0)
            logsumexp_flat = torch.logsumexp(u_flat_centered, dim=0).expand_as(u_flat_centered)
            
            if use_poly_loss:
                p_flat = torch.exp(log_p_flat)
                ce_term = -(q_flat.detach() * log_p_flat)
                poly_term = q_flat.detach() * poly_epsilon * (1.0 - p_flat)
                loss = (ce_term + poly_term).mean()
            else:
                loss = -(q_flat.detach() * log_p_flat).mean()
        
        # 记录分项 (Index Mode)
        term_logsumexp = logsumexp_flat.mean().detach()
        # alignment term: (q * u).mean() * G 近似
        term_alignment = (q_flat.detach() * u_flat_centered).mean().detach() * G
        u_center_val = (u_flat - u_flat_centered).mean().detach() if use_q_center else torch.tensor(0.0)
        
        # 如果是 scaled 变体
        if loss_variant == "scaled":
            loss = loss * grad_scale_factor
        
        # 用于 metrics
        adv_for_metrics = adv_flat
        u_for_metrics = u_flat
        q_for_metrics = q_flat
    
    # ============================================================
    # Step 6: 梯度裁剪（通过loss缩放）- 避免条件分支
    # ============================================================
    # 注意：不使用条件分支，因为loss值在不同rank上可能不同
    # 改为始终计算缩放因子，但当不需要裁剪时scale=1
    if grad_clip_value > 0:
        with torch.no_grad():
            # 使用max确保scale >= 1，这样不需要裁剪时不会改变loss
            scale = torch.clamp(loss.abs() / grad_clip_value, min=1.0)
        loss = loss / scale
    
    # ============================================================
    # Step 7: Logit Regularization (Z-Loss)
    # ============================================================
    # L_z = coef * (log Z)^2
    # 这能有效防止 Logits 整体漂移到无穷大
    logit_reg_loss = 0.0
    if logit_reg_coef > 0:
        # 使用 term_logsumexp (即 mean(logsumexp(u)))
        # 注意：这里我们惩罚的是 log Z 的平方
        logit_reg_loss = logit_reg_coef * (term_logsumexp ** 2)
        loss = loss + logit_reg_loss
    
    # ============================================================
    # Metrics
    # ============================================================
    with torch.no_grad():
        kl_val = log_ratio.abs().mean().item()
        
        # 诊断指标
        if loss_variant == "softmax" and not use_index_mode:
            p = F.softmax(u_for_metrics, dim=-1)
            p_minus_q_std = (p - q_for_metrics).std().item()
        else:
            p_minus_q_std = 0.0
        
        metrics = {
            "actor/ppo_kl": kl_val,
            "actor/pg_clipfrac": 0.0,
            "adpo/loss": loss.detach().item(),
            "adpo/advantage_mean": adv_for_metrics.mean().item(),
            "adpo/advantage_std": adv_for_metrics.std().item(),
            "adpo/anchored_score_mean": u_for_metrics.mean().item(),
            "adpo/anchored_score_std": u_for_metrics.std().item(),
            "adpo/effective_tau": float(effective_tau.item() if isinstance(effective_tau, torch.Tensor) else effective_tau),
            "adpo/p_minus_q_std": p_minus_q_std,
            "adpo/use_index_mode": use_index_mode,
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
