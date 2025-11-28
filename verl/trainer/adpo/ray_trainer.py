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
Inherits from RayPPOTrainer and uses on-policy anchoring (memory-efficient).
"""

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class RayADPOTrainer(RayPPOTrainer):
    """
    Distributed ADPO trainer using Ray for scalable reinforcement learning.
    
    ADPO (Anchored Direct Preference Optimization) uses an anchored distribution 
    p_θ(i|S) = softmax((s_i - s_anchor_i) / τ) instead of PPO-style clipping.
    
    This implementation uses on-policy mode only, where old_log_prob serves as
    the anchor. This is the most memory-efficient approach as it doesn't require
    maintaining a separate anchor model.
    
    The trainer inherits most functionality from RayPPOTrainer and only overrides
    ADPO-specific behavior:
    1. Uses "adpo" advantage estimator
    2. Uses "adpo" policy loss
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize ADPO trainer.
        
        All arguments are passed to RayPPOTrainer. ADPO-specific config should be
        in config.algorithm:
        - tau: Temperature for anchored softmax (default: 0.8)
        - use_adaptive_tau: Whether to use adaptive temperature (default: True)
        - adaptive_tau_alpha: Modulation strength for adaptive tau (default: 0.5)
        - adaptive_tau_min: Minimum tau value (default: 0.05)
        - beta_reward: Temperature for q computation (default: 0.5)
        - beta_anchor_kl: KL penalty coefficient (default: 0.0)
        - drop_all_failed_prompts: Whether to drop prompts with all 0 rewards (default: False)
        """
        super().__init__(*args, **kwargs)
        
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
            if not hasattr(algo_config, 'use_adaptive_tau'):
                algo_config.use_adaptive_tau = True
            if not hasattr(algo_config, 'adaptive_tau_alpha'):
                algo_config.adaptive_tau_alpha = 0.5
            if not hasattr(algo_config, 'adaptive_tau_min'):
                algo_config.adaptive_tau_min = 0.05
            if not hasattr(algo_config, 'beta_reward'):
                algo_config.beta_reward = 0.5
            if not hasattr(algo_config, 'beta_anchor_kl'):
                algo_config.beta_anchor_kl = 0.0
            if not hasattr(algo_config, 'drop_all_failed_prompts'):
                algo_config.drop_all_failed_prompts = False
            if not hasattr(algo_config, 'adaptive_tau_beta'):
                algo_config.adaptive_tau_beta = 0.5
            if not hasattr(algo_config, 'adaptive_tau_max'):
                algo_config.adaptive_tau_max = 1.0
            
            # Inject vocab_size for entropy normalization
            if not hasattr(algo_config, 'vocab_size'):
                tokenizer = kwargs.get('tokenizer')
                if tokenizer:
                    algo_config.vocab_size = tokenizer.vocab_size
                else:
                    # Fallback if tokenizer not in kwargs (though it should be)
                    algo_config.vocab_size = 32000  # Default fallback
                    print("Warning: Tokenizer not found in kwargs, using default vocab_size=32000 for ADPO entropy norm.")

        
        print(f"[ADPO] Initialized with tau={algo_config.tau}, "
              f"use_adaptive_tau={algo_config.use_adaptive_tau}")

