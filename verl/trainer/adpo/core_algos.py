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
        advantages: Computed advantages (shape: [batch_size])
        returns: Returns for value function (same as sequence-level rewards for ADPO)
    """
    with torch.no_grad():
        # Convert token-level rewards to sequence-level rewards
        # Sum over valid tokens - use in-place multiplication for memory efficiency
        sequence_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # Shape: [batch_size]
        
        num_generations = config.get("num_generations", 8)
        batch_size = sequence_rewards.shape[0]
        
        # Auto-truncate if not divisible
        if batch_size % num_generations != 0:
            valid_batch_size = (batch_size // num_generations) * num_generations
            if valid_batch_size == 0:
                returns = sequence_rewards
                return torch.zeros_like(returns), returns
            sequence_rewards = sequence_rewards[:valid_batch_size]
            batch_size = valid_batch_size
        
        num_prompts = batch_size // num_generations
        rewards_reshaped = sequence_rewards.view(num_prompts, num_generations)
        
        # Compute group-wise mean and advantages in-place where possible
        mean_rewards = rewards_reshaped.mean(dim=1, keepdim=True)
        advantages_reshaped = rewards_reshaped - mean_rewards
        
        # Optional: Scale by std if configured
        if config.get("scale_rewards", "group") == "group":
            std_rewards = rewards_reshaped.std(dim=1, keepdim=True)
            advantages_reshaped = advantages_reshaped / (std_rewards + 1e-8)
        
        advantages = advantages_reshaped.view(-1)
        returns = sequence_rewards
        
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
        advantages: Advantages (shape: [batch_size])
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
    # q = softmax(R / beta_reward)
    q_target = F.softmax(advantages_reshaped / beta_reward, dim=-1)
    
    # Compute adaptive temperature if enabled
    # Note: tau is treated as a hyperparameter, not a learned parameter,
    # so we compute it without gradients. Gradients flow through log_prob.
    if use_adaptive_tau:
        with torch.no_grad():
            # H(q) - entropy of target distribution
            entropy = -(q_target * torch.log(q_target + 1e-8)).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(num_generations, dtype=q_target.dtype, device=q_target.device))
            
            # tau(x) = max(tau_min, tau_base * (1 - alpha * H/H_max))
            adaptive_factor = 1.0 - adaptive_tau_alpha * (entropy / max_entropy)
            current_tau = torch.clamp(tau * adaptive_factor, min=adaptive_tau_min).unsqueeze(-1)
    else:
        current_tau = tau
    
    # Compute anchored scores: (s - s_anchor) / tau
    anchored_scores = (sequence_logps_reshaped - anchor_sequence_logps_reshaped) / current_tau
    
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
        }
        if use_adaptive_tau:
            metrics["adpo/mean_tau"] = current_tau.mean().item() if isinstance(current_tau, torch.Tensor) else current_tau
        if kl_loss is not None:
            metrics["adpo/kl_penalty"] = kl_loss.item()
        if valid_mask is not None:
            metrics["adpo/dropped_prompts"] = is_failed.sum().item()
    
    return loss, metrics

