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
ADPO Trainer with Ray-based single controller.
Inherits from RayPPOTrainer and overrides ADPO-specific logic.
"""

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class RayADPOTrainer(RayPPOTrainer):
    """
    Distributed ADPO trainer using Ray for scalable reinforcement learning.
    
    ADPO (Anchored Direct Preference Optimization) uses an anchored distribution 
    p_θ(i|S) = softmax((s_i - s_anchor_i) / τ) instead of PPO-style clipping.
    
    The trainer inherits most functionality from RayPPOTrainer and only overrides
    ADPO-specific behavior:
    1. Uses "adpo" advantage estimator
    2. Uses "adpo" policy loss
    3. Optionally maintains an anchor policy
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize ADPO trainer.
        
        All arguments are passed to RayPPOTrainer. ADPO-specific config should be
        in config.algorithm:
        - tau: Temperature for anchored softmax (default: 0.8)
        - anchor_update_mode: "on_policy", "fixed", "ema", or "kl_triggered" (default: "on_policy")
        - ema_alpha: EMA coefficient for anchor updates (default: 0.99)
        - kl_threshold: KL threshold for triggered updates (default: 0.1)
        - use_q_centering: Whether to center advantages (default: True)
        - beta_anchor_kl: KL penalty coefficient (default: 0.0)
        - use_adaptive_tau: Whether to use adaptive temperature (default: True)
        - adaptive_tau_alpha: Modulation strength for adaptive tau (default: 0.5)
        - adaptive_tau_min: Minimum tau value (default: 0.05)
        - beta_reward: Temperature for q computation (default: 0.5)
        - drop_all_failed_prompts: Whether to drop prompts with all 0 rewards (default: False)
        """
        super().__init__(*args, **kwargs)
        
        # ADPO-specific initialization
        self.anchor_update_count = 0
        self.kl_window = []
        self.kl_window_size = 10
        
        # Override advantage estimator to use ADPO
        if hasattr(self.config.algorithm, 'adv_estimator'):
            if self.config.algorithm.adv_estimator not in ['adpo', 'grpo']:
                print(f"Warning: ADPO trainer works best with 'adpo' or 'grpo' advantage estimator. "
                      f"Current: {self.config.algorithm.adv_estimator}")
        
        # Set default ADPO algorithm config if not present
        from omegaconf import open_dict
        
        algo_config = self.config.algorithm
        with open_dict(self.config):
            if not hasattr(algo_config, 'tau'):
                algo_config.tau = 0.8
            if not hasattr(algo_config, 'anchor_update_mode'):
                algo_config.anchor_update_mode = 'on_policy'
            if not hasattr(algo_config, 'ema_alpha'):
                algo_config.ema_alpha = 0.99
            if not hasattr(algo_config, 'kl_threshold'):
                algo_config.kl_threshold = 0.1
            if not hasattr(algo_config, 'use_q_centering'):
                algo_config.use_q_centering = True
            if not hasattr(algo_config, 'beta_anchor_kl'):
                algo_config.beta_anchor_kl = 0.0
            if not hasattr(algo_config, 'use_adaptive_tau'):
                algo_config.use_adaptive_tau = True
            if not hasattr(algo_config, 'adaptive_tau_alpha'):
                algo_config.adaptive_tau_alpha = 0.5
            if not hasattr(algo_config, 'adaptive_tau_min'):
                algo_config.adaptive_tau_min = 0.05
            if not hasattr(algo_config, 'beta_reward'):
                algo_config.beta_reward = 0.5
            if not hasattr(algo_config, 'drop_all_failed_prompts'):
                algo_config.drop_all_failed_prompts = False
        
        print(f"[ADPO] Initialized with tau={algo_config.tau}, "
              f"anchor_update_mode={algo_config.anchor_update_mode}, "
              f"use_adaptive_tau={algo_config.use_adaptive_tau}")

