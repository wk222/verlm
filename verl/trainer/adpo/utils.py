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

This module provides utility functions for ADPO training in on-policy mode.
"""

from typing import Dict, Any
import torch


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
    with torch.no_grad():
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
    tau: float,
    logger=None
) -> Dict[str, Any]:
    """
    Log ADPO-specific metrics.
    
    Args:
        metrics: Metrics dictionary to update
        anchor_kl: KL divergence from anchor
        tau: Current temperature value
        logger: Optional logger object
        
    Returns:
        Updated metrics dictionary
    """
    adpo_metrics = {
        "adpo/anchor_kl": anchor_kl,
        "adpo/tau": tau,
    }
    
    metrics.update(adpo_metrics)
    
    if logger is not None:
        logger.log(adpo_metrics)
    
    return metrics

