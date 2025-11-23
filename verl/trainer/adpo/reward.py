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
Reward functions for ADPO training, including good_accuracy reward.
"""

import re
from typing import Optional
from verl.trainer.ppo.reward import load_reward_manager as load_ppo_reward_manager


def load_reward_manager(config, tokenizer, num_examine=0, **kwargs):
    """
    Load reward manager for ADPO training.
    
    This is a wrapper around PPO's load_reward_manager that adds ADPO-specific
    reward functions if configured.
    
    Args:
        config: Configuration object
        tokenizer: Tokenizer
        num_examine: Number of examples to examine (for validation)
        **kwargs: Additional arguments for reward functions
        
    Returns:
        Reward manager function
    """
    # Use PPO's reward manager as base
    return load_ppo_reward_manager(config, tokenizer, num_examine, **kwargs)


def extract_boxed_answer(text: str) -> str:
    """从文本中提取 \\boxed{} 或 <answer> 标签中的答案。"""
    # 尝试提取 \\boxed{}
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 尝试提取 <answer> </answer>
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 如果都没有，返回最后一行非空内容
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines[-1] if lines else text.strip()


def good_accuracy(
    ngram_size: int = 4,
    max_penalty: float = -0.5,
    penalty_scale_factor: float = 0.1,
    **kwargs
):
    """
    计算组合奖励（使用 VERL 自带的 sympy 验证方法）。
    
    - 使用 VERL 的 prime_math.compute_score 进行数学等价性验证
    - 对错误答案应用 N-gram 重复惩罚
    - 不需要额外的依赖（latex2sympy2_extended）
    
    支持两种调用模式：
    1. 单个样本（NaiveRewardLoopManager）：
       - solution_str: 单个生成的文本
       - ground_truth: 单个答案
    2. 批处理（BatchRewardManager）：
       - completions: 多个生成的列表
       - solution: ground truth 列表

    参数:
        ngram_size: 用于计算重复惩罚的n-gram大小。
        max_penalty: 最大的（负数）重复惩罚值。
        penalty_scale_factor: 当答案错误时，应用于重复惩罚的缩放因子 (默认为 0.1)。
        **kwargs: Trainer 传入的数据集列。

    返回:
        float 或 list[float]: 奖励值（单个或列表）。
    """
    # 导入 VERL 自带的数学验证工具
    from verl.utils.reward_score import prime_math
    
    # 检测调用模式
    if "solution_str" in kwargs and "ground_truth" in kwargs:
        # 模式 1: NaiveRewardLoopManager (单个样本)
        content = kwargs["solution_str"]
        sol = kwargs["ground_truth"]
        
        # 提取模型答案
        extracted_answer = extract_boxed_answer(content)
        
        # 使用 VERL 的 prime_math 验证（基于 sympy）
        is_correct = False
        try:
            is_correct, format_correctness, _ = prime_math.compute_score(
                model_output=extracted_answer,
                ground_truth=sol
            )
        except Exception as e:
            print(f"验证失败: {e}, 回答: {extracted_answer}, 基准: {sol}")
            is_correct = False

        # === Reward Calculation ===
        final_reward = 0.0
        if is_correct:
            final_reward = 1.0
        else:
            # 计算重复惩罚
            repetition_penalty = 0.0
            words_in_content = content.lower().split()
            if content and len(words_in_content) >= ngram_size:
                try:
                    ngrams_set = set()
                    total_ngrams = 0
                    for ng in zip(*[words_in_content[i:] for i in range(ngram_size)]):
                        ngrams_set.add(ng)
                        total_ngrams += 1
                    if total_ngrams > 0:
                        scaling = 1.0 - (len(ngrams_set) / total_ngrams)
                        repetition_penalty = scaling * max_penalty
                except Exception as e:
                    print(f"计算重复惩罚时出错: {e}")
                    repetition_penalty = 0.0
            final_reward = 0.0 + penalty_scale_factor * repetition_penalty
            final_reward = min(final_reward, 0.0)

        return {"score": final_reward, "acc": 1.0 if is_correct else 0.0}
        
    elif "completions" in kwargs and "solution" in kwargs:
        # 模式 2: BatchRewardManager (批处理)
        completions = kwargs["completions"]
        solution = kwargs["solution"]
        
        if max_penalty > 0:
            raise ValueError(f"max_penalty {max_penalty} 应该是负数或零")

        final_rewards = []

        # --- Input Format Handling ---
        try:
            contents = [comp[0]["content"] for comp in completions]
        except (TypeError, IndexError, KeyError):
            if isinstance(completions, list) and all(isinstance(c, str) for c in completions):
                contents = completions
            else:
                raise ValueError("无法识别 completions 的格式 (既不是 list[str] 也不是 list[list[dict]])")
        
        if len(contents) != len(solution):
            raise ValueError(f"completions ({len(contents)}) 和 solution ({len(solution)}) 的数量必须匹配")

        for content, sol in zip(contents, solution):
            is_correct = False
            
            # 提取模型答案
            extracted_answer = extract_boxed_answer(content)
            
            # 使用 VERL 的 prime_math 验证（基于 sympy）
            try:
                is_correct, format_correctness, _ = prime_math.compute_score(
                    model_output=extracted_answer,
                    ground_truth=sol
                )
            except Exception as e:
                print(f"验证失败: {e}, 回答: {extracted_answer}, 基准: {sol}")
                is_correct = False

            # === Reward Calculation ===
            final_reward = 0.0
            if is_correct:
                final_reward = 1.0
            else:
                # 计算重复惩罚
                repetition_penalty = 0.0
                words_in_content = content.lower().split()
                if content and len(words_in_content) >= ngram_size:
                    try:
                        ngrams_set = set()
                        total_ngrams = 0
                        for ng in zip(*[words_in_content[i:] for i in range(ngram_size)]):
                            ngrams_set.add(ng)
                            total_ngrams += 1
                        if total_ngrams > 0:
                            scaling = 1.0 - (len(ngrams_set) / total_ngrams)
                            repetition_penalty = scaling * max_penalty
                    except Exception as e:
                        print(f"计算重复惩罚时出错: {e}")
                        repetition_penalty = 0.0
                final_reward = 0.0 + penalty_scale_factor * repetition_penalty
                final_reward = min(final_reward, 0.0)

            final_rewards.append(final_reward)

        return final_rewards
    else:
        raise ValueError("kwargs 必须包含 ('solution_str' 和 'ground_truth') 或 ('completions' 和 'solution')")

