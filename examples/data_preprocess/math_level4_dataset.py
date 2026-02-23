import os
import argparse
import pandas as pd
from datasets import load_dataset
from verl.utils.hdfs_io import copy, makedirs

def preprocess_dataset(local_save_dir):
    # Load dataset from HuggingFace
    dataset_name = "watermelonhjg/MATH-lighteval-level_4"
    print(f"Loading dataset {dataset_name}...")
    
    # This dataset usually has 'train' and 'test' splits
    dataset = load_dataset(dataset_name)
    
    # Define splits to process
    splits = ['train', 'test'] if 'test' in dataset else ['train']
    
    for split in splits:
        if split not in dataset:
            continue
            
        print(f"Processing split: {split}")
        ds = dataset[split]
        
        # Convert to pandas for easier manipulation
        df = ds.to_pandas()
        
        # Ensure required columns exist
        # VERL expects: data_source, prompt, ability, reward_model, extra_info
        
        # 1. data_source
        df['data_source'] = 'math'
        
        # 2. prompt
        if 'problem' in df.columns:
            system_prompt = (
                "You are a helpful AI Assistant that provides well-reasoned and detailed responses.\n"
                "First think about the necessary reasoning process as an internal monologue and then provide the user with the answer.\n"
                "Respond in the following format:\n"
                "<think>\n"
                "...\n"
                "</think>\n"
                "<answer>\n"
                "...\n"
                "</answer>\n"
            )
            
            def make_prompt(problem):
                return [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem}
                ]
            
            df['prompt'] = df['problem'].apply(make_prompt)
        else:
            raise ValueError(f"Column 'problem' not found in dataset. Columns: {df.columns}")
            
        # 3. ability
        df['ability'] = 'math'
        
        # 4. reward_model
        if 'solution' in df.columns:
            df['reward_model'] = df['solution'].apply(lambda x: {'style': 'rule', 'ground_truth': x})
        elif 'answer' in df.columns:
            df['reward_model'] = df['answer'].apply(lambda x: {'style': 'rule', 'ground_truth': x})
        else:
             raise ValueError(f"Column 'solution' or 'answer' not found. Columns: {df.columns}")
             
        # 5. extra_info
        df['extra_info'] = df.apply(lambda x: {'split': split}, axis=1)
        
        # Select columns
        df = df[['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']]
        
        # Save to parquet
        os.makedirs(local_save_dir, exist_ok=True)
        output_path = os.path.join(local_save_dir, f"{split}.parquet")
        df.to_parquet(output_path)
        print(f"Saved {split} split to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_save_dir", default="data/math_level4", help="Directory to save processed parquet files")
    args = parser.parse_args()
    
    preprocess_dataset(args.local_save_dir)
