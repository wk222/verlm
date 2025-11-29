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
import warnings
import logging
from typing import Optional
from verl.trainer.ppo.reward import load_reward_manager as load_ppo_reward_manager

# Suppress pylatexenc warnings about LaTeX macros like \frac
logging.getLogger('pylatexenc').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*macro.*failed its substitution.*')


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
    """从文本中提取 \\boxed{} 或 <answer> 标签中的答案。更鲁棒地处理嵌套括号。"""
    if not text:
        return ""
    
    # 方法1: 使用括号匹配找到最后一个 \boxed{} (处理嵌套)
    def find_last_boxed(s: str) -> str:
        idx = s.rfind("\\boxed")
        if idx < 0:
            idx = s.rfind("\\fbox")
            if idx < 0:
                return None
        
        i = idx
        left_brace_idx = None
        right_brace_idx = None
        num_left_braces_open = 0
        
        while i < len(s):
            if s[i] == "{":
                num_left_braces_open += 1
                if left_brace_idx is None:
                    left_brace_idx = i
            elif s[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        
        if left_brace_idx is not None and right_brace_idx is not None:
            return s[left_brace_idx + 1 : right_brace_idx].strip()
        return None
    
    # 尝试提取 \boxed{} (支持嵌套)
    boxed_answer = find_last_boxed(text)
    if boxed_answer:
        return boxed_answer
    
    # 尝试提取 <answer> </answer>
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 尝试匹配 "answer is X" 或 "the answer is X" 模式
    for pattern in [
        r'(?:the\s+)?answer\s+is[:\s]+([^\n\.]+)',
        r'(?:the\s+)?final\s+answer\s+is[:\s]+([^\n\.]+)',
        r'=\s*([\d\.\-\/\\\{\}\^\s]+)\s*$',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 如果都没有，返回最后一行非空内容
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines[-1] if lines else text.strip()


def good_accuracy(
    ngram_size: int = 4,
    max_penalty: float = -0.5,
    penalty_scale_factor: float = 0.5,
    length_reward_scale: float = 2000.0,
    length_reward_max: float = 0.1,
    repetition_threshold: float = 0.3,
    repetition_threshold_penalty: float = -0.5,
    **kwargs
):
    """
    计算组合奖励 (Improved Good Accuracy):
    1. 正确 = 1.0
    2. 错误 = 软长度奖励 (防止闭嘴) + 重复性惩罚 (防止废话)
    
    Args:
        ngram_size: N-gram 大小
        max_penalty: 最大惩罚值 (负数)
        penalty_scale_factor: 重复惩罚的缩放因子
        length_reward_scale: 长度奖励分母
        length_reward_max: 长度奖励上限
        repetition_threshold: 重复率阈值
        repetition_threshold_penalty: 超过阈值的固定惩罚
    """
    # 导入 VERL 自带的数学验证工具
    from verl.utils.reward_score import prime_math
    
    # 检测调用模式
    if "solution_str" in kwargs and "ground_truth" in kwargs:
        # 模式 1: NaiveRewardLoopManager (单个样本)
        content = kwargs["solution_str"]
        sol = kwargs["ground_truth"]
        
        # 从 ground_truth 中提取答案（可能是完整解答包含 \boxed{}）
        gt_answer = extract_boxed_answer(sol) if '\\boxed' in sol or '\\fbox' in sol else sol
        
        # 从模型输出中提取答案
        model_answer = extract_boxed_answer(content)
        
        # 使用 grade_answer 比较（更鲁棒）
        is_correct = False
        try:
            is_correct = prime_math.grade_answer(model_answer, gt_answer)
        except Exception as e:
            # 降级：直接字符串比较
            is_correct = model_answer.strip() == gt_answer.strip()

        # === Reward Calculation ===
        final_reward = 0.0
        if is_correct:
            final_reward = 1.0
        else:
            # A. 软长度奖励
            length_score = min(len(content) / length_reward_scale, length_reward_max)
            
            # B. 重复性惩罚
            repetition_penalty = 0.0
            hit_repetition_threshold = False
            
            words_in_content = content.lower().split()
            if content and len(words_in_content) >= ngram_size:
                try:
                    ngrams_set = set()
                    total_ngrams = 0
                    for ng in zip(*[words_in_content[i:] for i in range(ngram_size)]):
                        ngrams_set.add(ng)
                        total_ngrams += 1
                    if total_ngrams > 0:
                        ratio = len(ngrams_set) / total_ngrams
                        scaling = 1.0 - ratio # 1.0 means full repetition
                        
                        if scaling > repetition_threshold:
                            hit_repetition_threshold = True
                        else:
                            repetition_penalty = scaling * max_penalty
                except Exception as e:
                    print(f"计算重复惩罚时出错: {e}")
                    repetition_penalty = 0.0
            
            if hit_repetition_threshold:
                final_reward = repetition_threshold_penalty
            else:
                final_reward = length_score + repetition_penalty * penalty_scale_factor
                # 确保不超过 0 (虽然 length_score 是正的，但通常我们希望错误答案总分 <= 0 或者很小)
                # OPENR1 逻辑是 length_score + penalty。如果 length_score > penalty，可能是正分。
                # 这里保留 OPENR1 的原意：允许微小的正分鼓励输出，或者我们限制它。
                # 用户说“避免闭口”，所以微小正分是可以接受的。
                pass

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
            # 从 ground_truth 中提取答案（可能是完整解答包含 \boxed{}）
            gt_answer = extract_boxed_answer(sol) if '\\boxed' in sol or '\\fbox' in sol else sol
            
            # 从模型输出中提取答案
            model_answer = extract_boxed_answer(content)
            
            # 使用 grade_answer 比较（更鲁棒）
            is_correct = False
            try:
                is_correct = prime_math.grade_answer(model_answer, gt_answer)
            except Exception as e:
                # 降级：直接字符串比较
                is_correct = model_answer.strip() == gt_answer.strip()

            # === Reward Calculation ===
            final_reward = 0.0
            if is_correct:
                final_reward = 1.0
            else:
                # A. 软长度奖励
                length_score = min(len(content) / length_reward_scale, length_reward_max)
                
                # B. 重复性惩罚
                repetition_penalty = 0.0
                hit_repetition_threshold = False
                
                words_in_content = content.lower().split()
                if content and len(words_in_content) >= ngram_size:
                    try:
                        ngrams_set = set()
                        total_ngrams = 0
                        for ng in zip(*[words_in_content[i:] for i in range(ngram_size)]):
                            ngrams_set.add(ng)
                            total_ngrams += 1
                        if total_ngrams > 0:
                            ratio = len(ngrams_set) / total_ngrams
                            scaling = 1.0 - ratio
                            
                            if scaling > repetition_threshold:
                                hit_repetition_threshold = True
                            else:
                                repetition_penalty = scaling * max_penalty
                    except Exception as e:
                        print(f"计算重复惩罚时出错: {e}")
                        repetition_penalty = 0.0
                
                if hit_repetition_threshold:
                    final_reward = repetition_threshold_penalty
                else:
                    final_reward = length_score + repetition_penalty * penalty_scale_factor

            final_rewards.append(final_reward)

        return final_rewards
    else:
        raise ValueError("kwargs 必须包含 ('solution_str' 和 'ground_truth') 或 ('completions' 和 'solution')")

