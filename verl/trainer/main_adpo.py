# Copyright 2025 ADPO Algorithm
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
#
# This file integrates ADPO into the VERL framework (Apache 2.0, Bytedance Ltd.)
"""
Main entry point for ADPO training.
"""

import hydra
import ray

from verl.trainer.main_ppo import (
    TaskRunner as PPOTaskRunner,
    create_rl_dataset,
    create_rl_sampler,
)
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.adpo.ray_trainer import RayADPOTrainer
from verl.trainer.adpo.reward import load_reward_manager
# Import ADPO core algorithms to register the policy loss and advantage estimator
from verl.trainer.adpo import core_algos as adpo_core_algos  # noqa: F401
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="adpo_trainer", version_base=None)
def main(config):
    """Main entry point for ADPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    run_adpo(config)


def run_adpo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed ADPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed ADPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: Optional custom TaskRunner class.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(ADPOTaskRunner)

    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class ADPOTaskRunner(PPOTaskRunner):
    """
    Ray remote class for executing distributed ADPO training tasks.
    
    Inherits from PPOTaskRunner and overrides the run method to use
    RayADPOTrainer instead of RayPPOTrainer.
    """

    def run(self, config):
        """Execute the main ADPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the ADPO training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the ADPO training process.
        """
        from pprint import pprint
        import socket
        import os
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        from verl.utils.dataset.rl_dataset import collate_fn

        print(f"ADPOTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Add workers
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # Validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Load model
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate tokenizer and processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load reward manager
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        # Initialize resource pool manager
        resource_pool_manager = self.init_resource_pool_mgr(config)

        # Create datasets
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize ADPO trainer
        trainer = RayADPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

        # Initialize workers
        trainer.init_workers()

        # Start training
        trainer.fit()


if __name__ == "__main__":
    main()

