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

"""ADPO (Anchored Direct Preference Optimization) Trainer module."""

from .core_algos import adpo_policy_loss
from .ray_trainer import RayADPOTrainer
from .reward import load_reward_manager

__all__ = [
    "RayADPOTrainer",
    "adpo_policy_loss",
    "load_reward_manager",
]

