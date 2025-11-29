# Copyright 2025 ADPO Algorithm Author
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ADPO (Anchored Direct Preference Optimization) Algorithm
# Original paper: https://arxiv.org/abs/2510.18913
"""
Core functions to implement ADPO (Anchored Direct Preference Optimization) algorithms.

This implementation uses on-policy mode only, where old_log_prob serves as the anchor.
This is the most memory-efficient and commonly used configuration.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from verl.trainer.ppo.core_algos import register_adv_est, register_policy_loss


@register_adv_est("adpo")
def compute_adpo_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: DictConfig,
    index: Optional[torch.Tensor] = None,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute ADPO advantages using group-wise normalization.
    
    ADPO uses the same advantage computation as GRPO:
    advantages = rewards - mean(rewards_per_group)
    
    Args:
        token_level_rewards: Token-level rewards (shape: [batch_size, seq_len])
        response_mask: Response mask (shape: [batch_size, seq_len])
        config: Algorithm configuration
        index: Optional index array for grouping
        **kwargs: Additional arguments for compatibility
        
    Returns:
        advantages: Computed advantages (shape: [batch_size]) - sequence-level
        returns: Returns (shape: [batch_size]) - sequence-level
    """
    with torch.no_grad():
        # Convert token-level rewards to sequence-level rewards
        # Sum over valid tokens
        sequence_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # Shape: [batch_size]
        
        num_generations = config.get("num_generations", 8)
        batch_size = sequence_rewards.shape[0]
        
        # Store original batch size for padding later
        original_batch_size = batch_size
        
        # Auto-truncate if not divisible
        if batch_size % num_generations != 0:
            valid_batch_size = (batch_size // num_generations) * num_generations
            if valid_batch_size == 0:
                # Return sequence-level tensors filled with zeros
                return torch.zeros(original_batch_size, device=sequence_rewards.device, dtype=sequence_rewards.dtype), sequence_rewards
            sequence_rewards_truncated = sequence_rewards[:valid_batch_size]
            batch_size = valid_batch_size
        else:
            sequence_rewards_truncated = sequence_rewards
        
        num_prompts = batch_size // num_generations
        rewards_reshaped = sequence_rewards_truncated.view(num_prompts, num_generations)
        
        # Compute advantages based on configuration
        if config.get("use_kde_advantage", False):
            # KDE-based Advantage Estimation
            # Idea: Map rewards to their percentile in the estimated distribution (CDF),
            # then transform to logits: Adv = logit(CDF(r))
            
            # 1. Compute Bandwidth (h) using Scott's Rule
            # h = factor * 1.06 * std * N^(-1/5)
            std_rewards = rewards_reshaped.std(dim=1, keepdim=True)
            # Handle zero std (all rewards equal)
            std_rewards = torch.where(std_rewards < 1e-6, torch.ones_like(std_rewards), std_rewards)
            
            bandwidth_factor = config.get("kde_bandwidth_factor", 1.0)
            h = bandwidth_factor * 1.06 * std_rewards * (num_generations ** (-0.2))
            
            # 2. Compute Pairwise Differences for KDE
            # r_j: [B, N, 1], r_k: [B, 1, N]
            r_j = rewards_reshaped.unsqueeze(2)
            r_k = rewards_reshaped.unsqueeze(1)
            
            # u = (r_j - r_k) / h
            u = (r_j - r_k) / (h.unsqueeze(2) + 1e-8)
            
            # 3. Compute CDF using Gaussian Kernel
            # CDF(x) = 1/N * sum(Phi(u))
            # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
            cdf_contributions = 0.5 * (1 + torch.erf(u / 1.41421356))
            cdf = cdf_contributions.mean(dim=2) # [B, N]
            
            # 4. Logit Transformation
            # Clip to avoid infinity: [eps, 1-eps]
            epsilon = 1e-4
            cdf = cdf.clamp(epsilon, 1 - epsilon)
            advantages_reshaped = torch.log(cdf / (1 - cdf))
            
        elif config.get("scale_rewards", False):
            # Standard Normalized Advantage
            mean_rewards = rewards_reshaped.mean(dim=1, keepdim=True)
            std_rewards = rewards_reshaped.std(dim=1, keepdim=True)
            advantages_reshaped = (rewards_reshaped - mean_rewards) / (std_rewards + 1e-8)
            
        else:
            # Default: Centered Advantage (Reward - Mean)
            mean_rewards = rewards_reshaped.mean(dim=1, keepdim=True)
            advantages_reshaped = rewards_reshaped - mean_rewards
        
        # Flatten back to [valid_batch_size]
        advantages = advantages_reshaped.view(-1)
        
        # Flatten back to [valid_batch_size]
        advantages = advantages_reshaped.view(-1)
        returns = sequence_rewards_truncated
        
        # Pad back to original batch size if truncated
        if batch_size < original_batch_size:
            pad_size = original_batch_size - batch_size
            advantages = torch.cat([advantages, torch.zeros(pad_size, device=advantages.device, dtype=advantages.dtype)])
            returns = torch.cat([returns, sequence_rewards[batch_size:]])
        
    return advantages, returns


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
    ADPO policy loss: Anchored listwise cross-entropy (on-policy mode only).
    
    Uses on-policy mode where old_log_prob serves as the anchor policy.
    This is memory-efficient as it doesn't require maintaining a separate anchor model.
    
    Loss formulation:
    L = -E[q_target * log(p_anchored)]
    where p_anchored(i|S) = softmax((s_i - s_anchor_i) / tau)
    
    Args:
        old_log_prob: Anchor policy log probabilities (on-policy mode)
        log_prob: Current policy log probabilities (shape: [batch_size, seq_len])
        advantages: Advantages (shape: [batch_size]) - sequence-level
        response_mask: Mask for valid tokens (shape: [batch_size, seq_len])
        loss_agg_mode: How to aggregate loss (default: "mean")
        config: Algorithm configuration containing ADPO hyperparameters
        rollout_log_probs: Optional rollout log probs (unused in on-policy mode)
        
    Returns:
        loss: Scalar loss tensor
        metrics: Dictionary of metrics for logging
    """
    assert config is not None, "Config is required for ADPO loss"
    
    # Get ADPO config with defaults
    tau = config.get("tau", 0.8)
    num_generations = config.get("num_generations", 8)
    beta_reward = config.get("beta_reward", 0.5)
    use_adaptive_tau = config.get("use_adaptive_tau", True)
    adaptive_tau_alpha = config.get("adaptive_tau_alpha", 0.5)
    adaptive_tau_min = config.get("adaptive_tau_min", 0.05)
    beta_anchor_kl = config.get("beta_anchor_kl", 0.0)
    drop_all_failed_prompts = config.get("drop_all_failed_prompts", False)
    
    batch_size = advantages.shape[0]
    
    # Auto-truncate batch if not divisible by num_generations
    if batch_size % num_generations != 0:
        valid_batch_size = (batch_size // num_generations) * num_generations
        if valid_batch_size == 0:
            return torch.tensor(0.0, device=advantages.device, requires_grad=True), {}
        
        # Truncate all tensors
        advantages = advantages[:valid_batch_size]
        log_prob = log_prob[:valid_batch_size]
        old_log_prob = old_log_prob[:valid_batch_size]
        response_mask = response_mask[:valid_batch_size]
        batch_size = valid_batch_size
    
    num_prompts = batch_size // num_generations
    
    # Compute sequence-level log probs by summing over tokens
    # On-policy mode: use old_log_prob as anchor
    sequence_logps = (log_prob * response_mask).sum(dim=-1)
    anchor_sequence_logps = (old_log_prob * response_mask).sum(dim=-1)
    
    # Reshape for group-wise operations
    advantages_reshaped = advantages.view(num_prompts, num_generations)
    sequence_logps_reshaped = sequence_logps.view(num_prompts, num_generations)
    anchor_sequence_logps_reshaped = anchor_sequence_logps.view(num_prompts, num_generations)
    
    # Compute q_target: softmax of advantages
    if config.get("scale_rewards", False):
        # Normalize: (R - mean) / (std + eps)
        mean_adv = advantages_reshaped.mean(dim=1, keepdim=True)
        std_adv = advantages_reshaped.std(dim=1, keepdim=True)
        advantages_norm = (advantages_reshaped - mean_adv) / (std_adv + 1e-8)
        q_target = F.softmax(advantages_norm / beta_reward, dim=-1)
    else:
        # Use raw advantages (already centered)
        q_target = F.softmax(advantages_reshaped / beta_reward, dim=-1)
    
    # Compute adaptive temperature if enabled
    # Note: tau is treated as a hyperparameter, not a learned parameter,
    # so we compute it without gradients. Gradients flow through log_prob.
    if use_adaptive_tau:
        with torch.no_grad():
            # ============================================================
            # Smart Hybrid Adaptive Strategy
            # ============================================================
            
            # 1. Get Static Entropy (from Sampling Policy / old_log_prob)
            # Physical meaning: "Native Signal-to-Noise Ratio". If generation was messy, the problem is hard.
            # Role: Determines the base Uncertainty Factor (alpha term).
            token_entropy_static = -old_log_prob * response_mask
            mean_static = token_entropy_static.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1.0)
            
            # 2. Get Dynamic Entropy (from Current Policy / log_prob, detached)
            # Physical meaning: Model's "current" attitude towards this problem.
            # Role: Real-time monitoring for "blind confidence" (beta term).
            token_entropy_dynamic = -log_prob.detach() * response_mask
            mean_dynamic = token_entropy_dynamic.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1.0)

            # Normalization
            vocab_size = config.get("vocab_size", 32000)
            vocab_log = torch.log(torch.tensor(vocab_size, dtype=log_prob.dtype, device=log_prob.device))
            
            # Aggregate to Prompt Level (Group Mean)
            H_static = mean_static.view(num_prompts, num_generations).mean(dim=1) / vocab_log
            H_dynamic = mean_dynamic.view(num_prompts, num_generations).mean(dim=1) / vocab_log
            
            # Clamp entropies
            H_static = H_static.clamp(0, 1)
            H_dynamic = H_dynamic.clamp(0, 1)
            
            # 3. Prepare Rewards (R_norm)
            if "rewards" in kwargs:
                rewards = kwargs["rewards"]
                # Ensure rewards match batch size (in case of truncation)
                if rewards.shape[0] != batch_size:
                     rewards = rewards[:batch_size]
                
                rewards_grouped = rewards.view(num_prompts, num_generations)
                mean_rewards = rewards_grouped.mean(dim=1) # [B_prompts]
                
                # Min-max normalization within batch
                r_min, r_max = mean_rewards.min(), mean_rewards.max()
                if r_max > r_min + 1e-6:
                    normalized_reward = (mean_rewards - r_min) / (r_max - r_min)
                else:
                    normalized_reward = torch.full_like(mean_rewards, 0.5)
                normalized_reward = normalized_reward.clamp(0, 1)
            else:
                # Fallback if rewards not available
                normalized_reward = torch.full_like(H_static, 0.5)
            
            # ============================================================
            # 4. Smart Combination Formula
            # ============================================================
            
            # [A. Base Protection] Use Static Entropy
            # Logic: If data itself is low quality (high H_static), increase Tau globally to avoid overfitting noise.
            # Why not dynamic? Because H_dynamic decreases with training, we don't want to relax constraints just because model memorized noise.
            uncertainty_term = adaptive_tau_alpha * H_static
            
            # [B. Emergency Brake] Use Dynamic Entropy
            # Logic: If model becomes extremely confident "now" (low H_dynamic) but is wrong (high Error), punish immediately.
            # Why not static? Because old model might have been humble, failing to detect current collapse.
            current_confidence = (1.0 - H_dynamic).clamp(min=0)
            current_error = (1.0 - normalized_reward).clamp(min=0)
            
            # Default beta to 5.0 if not specified, as this is a strong penalty
            adaptive_tau_beta = config.get("adaptive_tau_beta", 5.0)
            penalty_term = adaptive_tau_beta * current_confidence * current_error
            
            # 5. Compute Final Tau
            adaptive_tau_max = config.get("adaptive_tau_max", 10.0)
            current_tau = tau * (1.0 + uncertainty_term + penalty_term)
            current_tau = torch.clamp(current_tau, min=adaptive_tau_min, max=adaptive_tau_max)
            
            # Broadcast to [B_prompts, 1]
            current_tau = current_tau.unsqueeze(-1)
    else:
        current_tau = tau
    
    # Compute anchored scores: (s - s_anchor) / tau
    anchored_scores = (sequence_logps_reshaped - anchor_sequence_logps_reshaped) / current_tau
    
    # Q-Centering (Numerical Stability Optimization)
    # u_bar = u - (q * u).sum()
    # This keeps logits centered and prevents drift, improving FP16 stability
    if config.get("use_q_centering", True):
        # q_target should be detached to act as a fixed weight
        centering_term = (q_target.detach() * anchored_scores).sum(dim=-1, keepdim=True)
        anchored_scores = anchored_scores - centering_term

    # ADPO listwise loss: cross-entropy
    log_p_anchored = F.log_softmax(anchored_scores, dim=-1)
    per_prompt_loss = -(q_target * log_p_anchored).sum(dim=-1)
    
    # Handle failed prompts if configured
    if drop_all_failed_prompts and "rewards" in kwargs:
        rewards = kwargs["rewards"].view(num_prompts, num_generations)
        is_failed = (rewards <= 0.0).all(dim=1)
        valid_mask = ~is_failed
        if valid_mask.sum() == 0:
            loss = torch.tensor(0.0, device=per_prompt_loss.device, requires_grad=True)
        else:
            loss = per_prompt_loss[valid_mask].mean()
    else:
        valid_mask = None
        loss = per_prompt_loss.mean()
    
    # Optional: Add KL penalty to anchor
    if beta_anchor_kl > 0:
        per_token_kl = torch.exp(old_log_prob - log_prob) - (old_log_prob - log_prob) - 1
        
        current_response_mask = response_mask
        if valid_mask is not None:
            valid_mask_expanded = valid_mask.repeat_interleave(num_generations)
            current_response_mask = response_mask * valid_mask_expanded.unsqueeze(1)
        
        kl_loss = (per_token_kl * current_response_mask).sum() / current_response_mask.sum().clamp(min=1.0)
        loss = loss + beta_anchor_kl * kl_loss
    else:
        kl_loss = None
    
    # Compute metrics (no gradients needed)
    with torch.no_grad():
        kl_val = (sequence_logps - anchor_sequence_logps).abs().mean().item()
        metrics = {
            "adpo/anchor_kl": kl_val,
            "adpo/loss": per_prompt_loss.mean().item(),
            "adpo/advantage_mean": advantages.mean().item(),
            "adpo/advantage_std": advantages.std().item(),
            "adpo/advantage_max": advantages.max().item(),
            "adpo/advantage_min": advantages.min().item(),
            "adpo/q_target_entropy": (-(q_target * torch.log(q_target + 1e-8)).sum(dim=-1)).mean().item(),
        }
        if use_adaptive_tau:
            metrics["adpo/mean_tau"] = current_tau.mean().item() if isinstance(current_tau, torch.Tensor) else current_tau
            metrics["adpo/entropy_static_mean"] = H_static.mean().item()
            metrics["adpo/entropy_dynamic_mean"] = H_dynamic.mean().item()
            metrics["adpo/reward_norm_mean"] = normalized_reward.mean().item()
            metrics["adpo/penalty_term_mean"] = penalty_term.mean().item()
        if kl_loss is not None:
            metrics["adpo/kl_penalty"] = kl_loss.item()
        if valid_mask is not None:
            metrics["adpo/dropped_prompts"] = is_failed.sum().item()
    
    return loss, metrics

