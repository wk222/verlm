import re

from verl.utils.reward_score.math_reward import compute_score

def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} or <answer> tags."""
    # Try \\boxed{}
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Try <answer> ... </answer>
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Fallback: return last line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines[-1] if lines else text.strip()

def compute_adpo_score(
    solution_str: str,
    ground_truth: str,
    ngram_size: int = 3,
    max_penalty: float = -0.5,
    penalty_scale_factor: float = 0.5,
    length_reward_scale: float = 2000.0,
    length_reward_max: float = 0.1,
    repetition_threshold: float = 0.3,
    repetition_threshold_penalty: float = -0.5
) -> float:
    """
    ADPO Math Reward with Soft Length and Repetition Penalty.
    
    Logic:
    1. Correct = 1.0 (uses verl's native math_reward.compute_score)
    2. Incorrect = Soft Length Reward + Repetition Penalty
       - Encourages outputting *something* (avoiding silence)
       - Penalizes repetitive loops (avoiding infinite CoT)
    """
    
    # 1. Check correctness
    # Extract answer first
    extracted_answer = extract_boxed_answer(solution_str)
    
    # compute_score returns 1.0 if correct, 0.0 otherwise
    # We pass the extracted answer as 'model_output' to prime_math (via math_reward)
    # Note: math_reward.compute_score expects the full string usually if it has logic to extract, 
    # but since we want to support <answer> tags which math_reward might not know, we pass extracted.
    # Wait, math_reward.compute_score calls last_boxed_only_string. 
    # If we pass "42", last_boxed_only_string("42") returns None (no boxed).
    # We should probably try to wrap it in boxed if it's not?
    # Actually, let's look at math_reward.compute_score again.
    # It calls last_boxed_only_string(solution_str).
    # If we pass just the number "42", it returns None.
    # So we must construct a string that looks like boxed if we extracted it from <answer>.
    
    # If extracted_answer is just "42", we make it "\\boxed{42}"
    if "\\boxed" not in extracted_answer:
        candidate_str = f"\\boxed{{{extracted_answer}}}"
    else:
        candidate_str = extracted_answer
        
    score = compute_score(candidate_str, ground_truth)
    
    if score == 1.0:
        return 1.0
        
    # 2. Calculate penalties for incorrect answers
    
    # A. Soft Length Reward: Encourage writing something, up to a limit
    # This helps avoid the "silence" failure mode where model outputs empty string to avoid negative reward
    length_score = min(len(solution_str) / length_reward_scale, length_reward_max)
    
    # B. Repetition Penalty: Prevent infinite loops
    repetition_penalty = 0.0
    hit_repetition_threshold = False
    
    if max_penalty < 0:
        words = solution_str.lower().split()
        if len(words) >= ngram_size:
            ngrams = set()
            total = 0
            for i in range(len(words) - ngram_size + 1):
                ng = tuple(words[i : i + ngram_size])
                ngrams.add(ng)
                total += 1
            
            if total > 0:
                # ratio is unique_ngrams / total_ngrams
                # scaling = 1 - ratio (0 = no repetition, 1 = full repetition)
                ratio = len(ngrams) / total
                scaling = 1.0 - ratio
                
                if scaling > repetition_threshold:
                    hit_repetition_threshold = True
                else:
                    repetition_penalty = scaling * max_penalty

    if hit_repetition_threshold:
        return float(repetition_threshold_penalty)
    else:
        return float(length_score + repetition_penalty * penalty_scale_factor)

# VERL expects 'compute_score' as the entry point
compute_score = compute_adpo_score
