"""
ADPO Configuration Example

This file demonstrates how to configure and use ADPO programmatically.
"""

from omegaconf import OmegaConf


def get_adpo_base_config():
    """Get base ADPO configuration."""
    config = OmegaConf.create({
        "algorithm": {
            "_target_": "verl.trainer.config.AlgoConfig",
            
            # Basic RL settings
            "gamma": 1.0,
            "lam": 1.0,
            
            # Use ADPO advantage estimator
            "adv_estimator": "adpo",
            "norm_adv_by_std_in_grpo": True,
            
            # Number of generations per prompt
            "num_generations": 8,
            
            # ========== ADPO Core Parameters ==========
            # Temperature for anchored softmax
            "tau": 0.8,
            
            # Anchor update mode: "on_policy", "fixed", "ema", "kl_triggered"
            "anchor_update_mode": "on_policy",
            
            # EMA coefficient (for ema mode)
            "ema_alpha": 0.99,
            
            # KL threshold (for kl_triggered mode)
            "kl_threshold": 0.1,
            
            # ========== Loss Function Parameters ==========
            # Whether to center advantages
            "use_q_centering": True,
            
            # KL penalty coefficient for anchor KL
            "beta_anchor_kl": 0.0,
            
            # Temperature for q computation
            "beta_reward": 0.5,
            
            # Whether to drop failed prompts
            "drop_all_failed_prompts": False,
            
            # ========== Adaptive Temperature ==========
            # Enable adaptive tau
            "use_adaptive_tau": True,
            
            # Modulation strength
            "adaptive_tau_alpha": 0.5,
            
            # Minimum tau value
            "adaptive_tau_min": 0.05,
            
            # ========== Reward Scaling ==========
            "scale_rewards": "group",  # "batch", "group", or "none"
            
            # KL penalty settings (optional)
            "use_kl_in_reward": False,
            "kl_penalty": "kl",
            "kl_ctrl": {
                "_target_": "verl.trainer.config.KLControlConfig",
                "type": "fixed",
                "kl_coef": 0.001,
            },
        },
        
        "trainer": {
            "balance_batch": True,
            "total_epochs": 30,
            "project_name": "adpo_project",
            "experiment_name": "adpo_experiment",
            "logger": ["console", "wandb"],
            "nnodes": 1,
            "n_gpus_per_node": 8,
            "save_freq": 1,
        },
        
        "custom_reward_function": {
            "path": None,  # Set to your reward function file path
            "name": "compute_score",
        },
        
        "reward_model": {
            "enable": False,
            "reward_kwargs": {},
        },
    })
    
    return config


def get_adpo_on_policy_config():
    """ADPO configuration with on-policy anchor (like GRPO)."""
    config = get_adpo_base_config()
    
    with OmegaConf.open_dict(config):
        config.algorithm.anchor_update_mode = "on_policy"
        config.algorithm.tau = 0.8
        config.trainer.experiment_name = "adpo_on_policy"
    
    return config


def get_adpo_fixed_anchor_config():
    """ADPO configuration with fixed anchor (standard ADPO)."""
    config = get_adpo_base_config()
    
    with OmegaConf.open_dict(config):
        config.algorithm.anchor_update_mode = "fixed"
        config.algorithm.tau = 1.0
        config.trainer.experiment_name = "adpo_fixed_anchor"
    
    return config


def get_adpo_ema_config(ema_alpha=0.99):
    """ADPO configuration with EMA anchor updates."""
    config = get_adpo_base_config()
    
    with OmegaConf.open_dict(config):
        config.algorithm.anchor_update_mode = "ema"
        config.algorithm.ema_alpha = ema_alpha
        config.algorithm.tau = 0.8
        config.trainer.experiment_name = f"adpo_ema_alpha{int(ema_alpha*100)}"
    
    return config


def get_adpo_kl_triggered_config(kl_threshold=0.1):
    """ADPO configuration with KL-triggered anchor updates."""
    config = get_adpo_base_config()
    
    with OmegaConf.open_dict(config):
        config.algorithm.anchor_update_mode = "kl_triggered"
        config.algorithm.kl_threshold = kl_threshold
        config.algorithm.tau = 0.8
        config.trainer.experiment_name = f"adpo_kl_thresh{kl_threshold}"
    
    return config


def get_adpo_with_good_accuracy_config():
    """ADPO configuration with good_accuracy reward function."""
    config = get_adpo_base_config()
    
    with OmegaConf.open_dict(config):
        config.algorithm.anchor_update_mode = "on_policy"
        config.algorithm.tau = 0.8
        config.algorithm.drop_all_failed_prompts = True
        
        # Configure good_accuracy reward
        config.custom_reward_function.path = "verl/trainer/adpo/reward.py"
        config.custom_reward_function.name = "good_accuracy"
        
        config.reward_model.reward_kwargs = {
            "ngram_size": 4,
            "max_penalty": -0.5,
            "penalty_scale_factor": 0.1,
        }
        
        config.trainer.experiment_name = "adpo_good_accuracy"
    
    return config


def example_usage():
    """Example of how to use these configurations."""
    # Choose a configuration
    config = get_adpo_on_policy_config()
    # config = get_adpo_fixed_anchor_config()
    # config = get_adpo_ema_config(ema_alpha=0.99)
    # config = get_adpo_kl_triggered_config(kl_threshold=0.1)
    # config = get_adpo_with_good_accuracy_config()
    
    # Customize as needed
    with OmegaConf.open_dict(config):
        config.algorithm.num_generations = 16  # Increase generations
        config.algorithm.tau = 0.5  # Lower temperature
        config.trainer.total_epochs = 50  # More epochs
    
    # Print configuration
    print("ADPO Configuration:")
    print(OmegaConf.to_yaml(config))
    
    return config


if __name__ == "__main__":
    # Example: Generate and print configuration
    config = example_usage()
    
    # You can then use this config with:
    # from verl.trainer.main_adpo import run_adpo
    # run_adpo(config)

