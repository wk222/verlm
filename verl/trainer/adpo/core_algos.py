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
"""

from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from verl.trainer.ppo.core_algos import register_adv_est, register_policy_loss, AdvantageEstimator
from verl.utils import group_mean_std


@register_adv_est("adpo")
def compute_adpo_advantages(data, config):
    """
    Compute ADPO advantages using group-wise normalization.
    
    ADPO uses the same advantage computation as GRPO:
    advantages = rewards - mean(rewards_per_group)
    
    Args:
        data: DataProto containing rewards
        config: Algorithm configuration
        
    Returns:
        data: Updated DataProto with advantages
    """
    rewards = data.batch["rewards"]  # Shape: [batch_size]
    num_generations = config.algorithm.get("num_generations", 8)
    
    # Reshape to [num_prompts, num_generations]
    batch_size = rewards.shape[0]
    num_prompts = batch_size // num_generations
    rewards_reshaped = rewards.view(num_prompts, num_generations)
    
    # Compute group-wise mean
    mean_rewards = rewards_reshaped.mean(dim=1, keepdim=True)
    
    # Compute advantages
    advantages_reshaped = rewards_reshaped - mean_rewards
    advantages = advantages_reshaped.view(-1)
    
    # Optional: Scale by std if configured
    if config.algorithm.get("scale_rewards", "group") == "group":
        std_rewards = rewards_reshaped.std(dim=1, keepdim=True)
        advantages_reshaped = advantages_reshaped / (std_rewards + 1e-8)
        advantages = advantages_reshaped.view(-1)
    
    data.batch["advantages"] = advantages
    return data


@register_policy_loss("adpo")
def adpo_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "mean",
    config: Optional[DictConfig] = None,
    rollout_log_probs: Optional[torch.Tensor] = None,
    anchor_log_prob: Optional[torch.Tensor] = None,
    **kwargs
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    ADPO policy loss: Anchored listwise cross-entropy.
    
    Unlike GRPO's PPO-style clipping, ADPO uses:
    L = -E[q_target * log(p_anchored)]
    where p_anchored(i|S) = softmax((s_i - s_anchor_i) / tau)
    
    Args:
        old_log_prob: Old policy log probabilities (not used in ADPO, kept for compatibility)
        log_prob: Current policy log probabilities (shape: [batch_size, seq_len])
        advantages: Advantages (shape: [batch_size])
        response_mask: Mask for valid tokens (shape: [batch_size, seq_len])
        loss_agg_mode: How to aggregate loss (default: "mean")
        config: Algorithm configuration containing ADPO hyperparameters
        rollout_log_probs: Optional rollout log probs for importance sampling
        anchor_log_prob: Anchor policy log probabilities (shape: [batch_size, seq_len])
        
    Returns:
        loss: Scalar loss tensor
        metrics: Dictionary of metrics for logging
    """
    assert config is not None, "Config is required for ADPO loss"
    
    # Get ADPO config
    tau = config.algorithm.get("tau", 0.8)
    num_generations = config.algorithm.get("num_generations", 8)
    use_q_centering = config.algorithm.get("use_q_centering", True)
    beta_anchor_kl = config.algorithm.get("beta_anchor_kl", 0.0)
    use_adaptive_tau = config.algorithm.get("use_adaptive_tau", True)
    adaptive_tau_alpha = config.algorithm.get("adaptive_tau_alpha", 0.5)
    adaptive_tau_min = config.algorithm.get("adaptive_tau_min", 0.05)
    beta_reward = config.algorithm.get("beta_reward", 0.5)
    drop_all_failed_prompts = config.algorithm.get("drop_all_failed_prompts", False)
    
    # Compute sequence-level log probs by summing over tokens
    sequence_logps = (log_prob * response_mask).sum(dim=-1)  # [batch_size]
    
    # Use anchor_log_prob if provided, otherwise use old_log_prob (on-policy mode)
    if anchor_log_prob is not None:
        anchor_sequence_logps = (anchor_log_prob * response_mask).sum(dim=-1)
    else:
        # On-policy mode: use old_log_prob as anchor
        anchor_sequence_logps = (old_log_prob * response_mask).sum(dim=-1)
    
    # Reshape to [num_prompts, num_generations]
    batch_size = advantages.shape[0]
    num_prompts = batch_size // num_generations
    
    # Auto-truncate batch if not divisible by num_generations
    if batch_size % num_generations != 0:
        valid_batch_size = (batch_size // num_generations) * num_generations
        if valid_batch_size == 0:
            return torch.tensor(0.0, device=advantages.device, requires_grad=True), {}
        
        # Truncate all tensors
        advantages = advantages[:valid_batch_size]
        sequence_logps = sequence_logps[:valid_batch_size]
        anchor_sequence_logps = anchor_sequence_logps[:valid_batch_size]
        response_mask = response_mask[:valid_batch_size]
        log_prob = log_prob[:valid_batch_size]
        if anchor_log_prob is not None:
            anchor_log_prob = anchor_log_prob[:valid_batch_size]
        
        batch_size = valid_batch_size
        num_prompts = batch_size // num_generations
    
    # Reshape
    advantages_reshaped = advantages.view(num_prompts, num_generations)
    sequence_logps_reshaped = sequence_logps.view(num_prompts, num_generations)
    anchor_sequence_logps_reshaped = anchor_sequence_logps.view(num_prompts, num_generations)
    
    # Compute q_target with explicit normalization
    # Normalize advantages: R_norm = (R - mean) / (std + eps)
    mean_adv = advantages_reshaped.mean(dim=1, keepdim=True)
    std_adv = advantages_reshaped.std(dim=1, keepdim=True)
    advantages_norm = (advantages_reshaped - mean_adv) / (std_adv + 1e-8)
    
    # Calculate q = softmax(R_norm / beta_reward)
    q_logits = advantages_norm / beta_reward
    q_target = F.softmax(q_logits, dim=-1)
    
    # Adaptive Temperature Scaling
    if use_adaptive_tau:
        # H(q)
        entropy = -(q_target * torch.log(q_target + 1e-8)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(num_generations, dtype=q_target.dtype, device=q_target.device))
        
        # tau(x) = max(tau_min, tau_base * (1 - alpha * H/H_max))
        adaptive_factor = 1.0 - adaptive_tau_alpha * (entropy / max_entropy)
        current_tau = tau * adaptive_factor
        current_tau = torch.clamp(current_tau, min=adaptive_tau_min)
        
        # Broadcast: [num_prompts] -> [num_prompts, num_generations]
        current_tau = current_tau.unsqueeze(-1)
    else:
        current_tau = tau
    
    # Compute anchored scores: (s - s_anchor) / tau
    anchored_scores = (sequence_logps_reshaped - anchor_sequence_logps_reshaped) / current_tau
    
    # ADPO listwise loss: cross-entropy
    log_p_anchored = F.log_softmax(anchored_scores, dim=-1)
    per_prompt_loss = -(q_target * log_p_anchored).sum(dim=-1)
    
    # Drop failed prompts if requested
    valid_mask = None
    if drop_all_failed_prompts and "rewards" in kwargs:
        rewards = kwargs["rewards"].view(num_prompts, num_generations)
        is_failed = (rewards <= 0.0).all(dim=1)
        if is_failed.any():
            valid_mask = ~is_failed
            if valid_mask.sum() == 0:
                loss = torch.tensor(0.0, device=per_prompt_loss.device, requires_grad=True)
            else:
                loss = per_prompt_loss[valid_mask].mean()
        else:
            loss = per_prompt_loss.mean()
    else:
        loss = per_prompt_loss.mean()
    
    # Optional: Add KL penalty
    kl_loss = torch.tensor(0.0, device=loss.device)
    if beta_anchor_kl > 0:
        if anchor_log_prob is not None:
            per_token_kl = (
                torch.exp(anchor_log_prob - log_prob) -
                (anchor_log_prob - log_prob) - 1
            )
        else:
            per_token_kl = (
                torch.exp(old_log_prob - log_prob) -
                (old_log_prob - log_prob) - 1
            )
        
        # Mask KL for dropped prompts
        current_response_mask = response_mask
        if valid_mask is not None:
            valid_mask_expanded = valid_mask.repeat_interleave(num_generations)
            current_response_mask = response_mask * valid_mask_expanded.unsqueeze(1)
        
        kl_loss = (per_token_kl * current_response_mask).sum() / current_response_mask.sum().clamp(min=1.0)
        loss = loss + beta_anchor_kl * kl_loss
    
    # Compute metrics
    with torch.no_grad():
        kl_val = (sequence_logps - anchor_sequence_logps).abs().mean().item()
        metrics = {
            "adpo/anchor_kl": kl_val,
            "adpo/loss": per_prompt_loss.mean().item(),
        }
        if use_adaptive_tau:
            metrics["adpo/mean_tau"] = current_tau.mean().item() if isinstance(current_tau, torch.Tensor) else current_tau
        if beta_anchor_kl > 0:
            metrics["adpo/kl_penalty"] = kl_loss.item()
        if valid_mask is not None:
            metrics["adpo/dropped_prompts"] = is_failed.sum().item()
    
    return loss, metrics

