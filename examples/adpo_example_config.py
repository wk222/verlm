"""
ADPO Configuration Example

This file demonstrates how to configure and use ADPO programmatically.
ADPO uses on-policy anchoring (old_log_prob as anchor) for memory efficiency.
"""

from omegaconf import OmegaConf


def get_adpo_base_config():
    """
    Get base ADPO configuration.
    
    Uses on-policy anchoring where old_log_prob serves as the anchor.
    This is the most memory-efficient approach as it doesn't require
    maintaining a separate anchor model.
    """
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
            
            # ========== Loss Function Parameters ==========
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
    """ADPO configuration with on-policy anchor (default, memory-efficient)."""
    config = get_adpo_base_config()
    
    with OmegaConf.open_dict(config):
        config.algorithm.tau = 0.8
        config.trainer.experiment_name = "adpo_on_policy"
    
    return config


def get_adpo_with_good_accuracy_config():
    """ADPO configuration with good_accuracy reward function."""
    config = get_adpo_base_config()
    
    with OmegaConf.open_dict(config):
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


def get_adpo_memory_optimized_config():
    """ADPO configuration optimized for memory efficiency."""
    config = get_adpo_base_config()
    
    with OmegaConf.open_dict(config):
        # Smaller batch with gradient accumulation
        config.algorithm.tau = 0.8
        config.algorithm.use_adaptive_tau = True
        
        # Optimize memory
        config.trainer.gradient_checkpointing = True
        config.trainer.bf16 = True
        
        config.trainer.experiment_name = "adpo_memory_optimized"
    
    return config


def example_usage():
    """Example of how to use these configurations."""
    # Choose a configuration
    config = get_adpo_on_policy_config()
    # config = get_adpo_with_good_accuracy_config()
    # config = get_adpo_memory_optimized_config()
    
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

