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
Utility functions for ADPO training.
"""

from typing import Dict, Any
import torch


def update_anchor_policy_ema(anchor_model, current_model, alpha: float = 0.99):
    """
    Update anchor policy using Exponential Moving Average (EMA).
    
    Formula: anchor = alpha * anchor + (1-alpha) * current
    
    Args:
        anchor_model: The anchor policy model
        current_model: The current policy model
        alpha: EMA coefficient (higher = more stable anchor)
    """
    with torch.no_grad():
        for anchor_param, current_param in zip(
            anchor_model.parameters(),
            current_model.parameters()
        ):
            anchor_param.data.mul_(alpha).add_((1 - alpha) * current_param.data)


def update_anchor_policy_hard_copy(anchor_model, current_model):
    """
    Update anchor policy by hard copying current policy weights.
    
    Args:
        anchor_model: The anchor policy model
        current_model: The current policy model
    """
    with torch.no_grad():
        for anchor_param, current_param in zip(
            anchor_model.parameters(),
            current_model.parameters()
        ):
            anchor_param.data.copy_(current_param.data)


def should_update_anchor_kl_triggered(
    kl_window: list,
    kl_threshold: float,
    window_size: int = 10
) -> bool:
    """
    Check if anchor policy should be updated based on KL threshold.
    
    Args:
        kl_window: List of recent KL divergence values
        kl_threshold: Threshold for triggering update
        window_size: Size of the KL window
        
    Returns:
        True if anchor should be updated
    """
    if len(kl_window) < window_size:
        return False
    
    mean_kl = sum(kl_window) / len(kl_window)
    return mean_kl > kl_threshold


def compute_adaptive_tau(
    q_target: torch.Tensor,
    base_tau: float,
    alpha: float = 0.5,
    tau_min: float = 0.05,
    num_generations: int = 8
) -> torch.Tensor:
    """
    Compute adaptive temperature based on target distribution entropy.
    
    Formula: tau(x) = max(tau_min, base_tau * (1 - alpha * H/H_max))
    
    Args:
        q_target: Target distribution [num_prompts, num_generations]
        base_tau: Base temperature value
        alpha: Modulation strength
        tau_min: Minimum tau value
        num_generations: Number of generations per prompt
        
    Returns:
        Adaptive tau values [num_prompts]
    """
    # Compute entropy H(q)
    entropy = -(q_target * torch.log(q_target + 1e-8)).sum(dim=-1)
    
    # Maximum entropy for uniform distribution
    max_entropy = torch.log(torch.tensor(num_generations, dtype=q_target.dtype, device=q_target.device))
    
    # Compute adaptive factor
    adaptive_factor = 1.0 - alpha * (entropy / max_entropy)
    
    # Compute tau
    tau = base_tau * adaptive_factor
    tau = torch.clamp(tau, min=tau_min)
    
    return tau


def log_adpo_metrics(
    metrics: Dict[str, Any],
    anchor_kl: float,
    anchor_update_count: int,
    tau: float,
    logger=None
):
    """
    Log ADPO-specific metrics.
    
    Args:
        metrics: Metrics dictionary
        anchor_kl: KL divergence from anchor
        anchor_update_count: Number of anchor updates
        tau: Current temperature value
        logger: Optional logger object
    """
    adpo_metrics = {
        "adpo/anchor_kl": anchor_kl,
        "adpo/anchor_update_count": anchor_update_count,
        "adpo/tau": tau,
    }
    
    metrics.update(adpo_metrics)
    
    if logger is not None:
        logger.log(adpo_metrics)
    
    return metrics

