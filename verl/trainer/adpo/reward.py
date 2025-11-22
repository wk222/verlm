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


def is_pure_numerical(text: str) -> bool:
    """检查字符串是否完全由一个简单的整数或小数构成。"""
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not text:
        return False
    pattern = r"^[+-]?(\d+(\.\d*)?|\.\d+)$"
    return bool(re.fullmatch(pattern, text))


def good_accuracy(
    ngram_size: int = 4,
    max_penalty: float = -0.5,
    penalty_scale_factor: float = 0.1,
    **kwargs
):
    """
    计算组合奖励 (版本：强制使用原始逻辑，纯数字答案尝试包装后处理)。
    - **所有情况**都尝试使用原始的 parse/verify 逻辑。
    - 如果基准答案是纯数字，会先用花括号 {} 包装后再传入 parse。
    - **警告**: 此方法的准确性高度依赖于 parse/verify 处理包装后数字的能力，
               以及 verify 进行跨格式数值等价比较的能力。请务必测试！

    参数:
        ngram_size: 用于计算重复惩罚的n-gram大小。
        max_penalty: 最大的（负数）重复惩罚值。
        penalty_scale_factor: 当答案错误时，应用于重复惩罚的缩放因子 (默认为 0.1)。
        **kwargs: GRPOTrainer 传入的其他数据集列, 必须包含 "completions" 和 "solution"。

    返回:
        list[float]: 每个 completion 对应的奖励值列表。
    """
    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
    except ImportError:
        raise ImportError(
            "good_accuracy reward requires latex2sympy2_extended and math_verify. "
            "Install with: pip install latex2sympy2_extended math_verify"
        )
    
    if "completions" not in kwargs or "solution" not in kwargs:
        raise ValueError("kwargs 必须包含 'completions' 和 'solution'")

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

    # --- N-gram Helper ---
    def zipngram(text: str, n: int):
        words = text.lower().split()
        if len(words) < n:
            return []
        return zip(*[words[i:] for i in range(n)])

    for content, sol in zip(contents, solution):
        is_correct = False
        processed_sol = sol  # 默认使用原始 sol

        # --- 核心修改：如果是纯数字，用花括号包装 ---
        if is_pure_numerical(sol):
            processed_sol = f"{{{sol.strip()}}}"  # 例如 "0.5" -> "{0.5}"
            print(f"Info: Wrapping numerical solution '{sol}' to '{processed_sol}' for parse/verify.")

        # --- 统一使用原始的 parse/verify 逻辑 ---
        try:
            # 使用 processed_sol (可能是原始的，可能是包装后的)
            gold_parsed = parse(
                processed_sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()]
            )

            if len(gold_parsed) != 0:
                # 解析模型回答
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False
                        )
                    ],
                    extraction_mode="first_match"
                )
                try:
                    # 使用原始 verify 函数进行比较
                    is_correct = verify(answer_parsed, gold_parsed)
                except Exception as verify_err:
                    print(f"验证失败 (统一逻辑): {verify_err}, 回答: {answer_parsed}, "
                          f"基准: {gold_parsed} (来自: '{processed_sol}')")
                    is_correct = False
            else:
                # 如果 parse(processed_sol) 失败
                print(f"警告 (统一逻辑): 无法解析处理后的基准答案 '{processed_sol}' "
                      f"(来自原始: '{sol}'), 视为正确。")
                is_correct = True
        except Exception as parse_err:
            print(f"解析失败 (统一逻辑): {parse_err}, 回答: {content}, "
                  f"处理后基准: {processed_sol} (来自原始: {sol})")
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

