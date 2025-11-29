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
        
        # Compute group-wise mean and advantages
        mean_rewards = rewards_reshaped.mean(dim=1, keepdim=True)
        advantages_reshaped = rewards_reshaped - mean_rewards
        
        # Optional: Scale by std if configured
        if config.get("scale_rewards", "group") == "group":
            std_rewards = rewards_reshaped.std(dim=1, keepdim=True)
            advantages_reshaped = advantages_reshaped / (std_rewards + 1e-8)
        
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
    
    # Compute q_target: normalized softmax of advantages
    # Normalize: (R - mean) / (std + eps)
    mean_adv = advantages_reshaped.mean(dim=1, keepdim=True)
    std_adv = advantages_reshaped.std(dim=1, keepdim=True)
    advantages_norm = (advantages_reshaped - mean_adv) / (std_adv + 1e-8)
    
    # q = softmax(R_norm / beta_reward)
    q_target = F.softmax(advantages_norm / beta_reward, dim=-1)
    
    # Compute adaptive temperature if enabled
    # Note: tau is treated as a hyperparameter, not a learned parameter,
    # so we compute it without gradients. Gradients flow through log_prob.
    if use_adaptive_tau:
        with torch.no_grad():
            # ========================================
            # Smooth Hybrid Adaptive Temperature Scaling
            # ========================================
            # Formula: τ = τ_base × (1 + α·H_norm + β·(1-H_norm)·(1-R_norm))
            
            # 1. Estimate Model Entropy (H) using -log_prob of sampled tokens
            # H ≈ -1/T * sum(log p(x_t))
            # This is a proxy for model uncertainty (on-policy approximation)
            token_entropy = -log_prob * response_mask
            mean_token_entropy = token_entropy.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1.0) # [B]
            
            # Reshape to groups
            mean_token_entropy_grouped = mean_token_entropy.view(num_prompts, num_generations).mean(dim=1) # [B_prompts]
            
            # Normalize Entropy (H_norm)
            # H_norm = H / log(vocab_size)
            vocab_size = config.get("vocab_size", 32000)
            max_token_entropy = torch.log(torch.tensor(vocab_size, dtype=mean_token_entropy.dtype, device=mean_token_entropy.device))
            normalized_entropy = (mean_token_entropy_grouped / max_token_entropy).clamp(0, 1) # [B_prompts]
            
            # 2. Normalize Rewards (R_norm)
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
                # Fallback if rewards not available (should not happen in standard loop)
                normalized_reward = torch.full_like(normalized_entropy, 0.5)
            
            # 3. Compute Terms
            # confidence = 1 - H_norm (high when model is certain)
            # error = 1 - R_norm (high when rewards are low)
            confidence = (1.0 - normalized_entropy).clamp(min=0)
            error = (1.0 - normalized_reward).clamp(min=0)
            
            # Uncertainty term: protects against high entropy (confused model)
            uncertainty_term = adaptive_tau_alpha * normalized_entropy
            
            # Penalty term: punishes confident but wrong predictions (arrogant idiot)
            adaptive_tau_beta = config.get("adaptive_tau_beta", 0.5)
            penalty_term = adaptive_tau_beta * confidence * error
            
            # 4. Final Adaptive Tau
            adaptive_tau_max = config.get("adaptive_tau_max", 5.0)
            current_tau = tau * (1.0 + uncertainty_term + penalty_term)
            current_tau = torch.clamp(current_tau, min=adaptive_tau_min, max=adaptive_tau_max)
            
            # Broadcast to [B_prompts, 1] to match sequence_logps_reshaped [B_prompts, num_generations]
            current_tau = current_tau.unsqueeze(-1)
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
            metrics["adpo/entropy_norm_mean"] = normalized_entropy.mean().item()
            metrics["adpo/reward_norm_mean"] = normalized_reward.mean().item()
            metrics["adpo/penalty_term_mean"] = penalty_term.mean().item()
        if kl_loss is not None:
            metrics["adpo/kl_penalty"] = kl_loss.item()
        if valid_mask is not None:
            metrics["adpo/dropped_prompts"] = is_failed.sum().item()
    
    return loss, metrics

