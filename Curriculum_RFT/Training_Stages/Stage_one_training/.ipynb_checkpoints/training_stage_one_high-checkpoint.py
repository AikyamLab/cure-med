#!/usr/bin/env python3
"""
GRPO Training Script for Multilingual Medical Reasoning

"""
# ============================================================================
# IMPORTS AND ENVIRONMENT SETUP
# ============================================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append('..')
sys.path.append('.')
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainerCallback
import transformers
from trl import GRPOTrainer, GRPOConfig
import wandb
# WandB setup
wandb_api_key = "........"
if len(wandb_api_key) != 40:
    raise ValueError(f"Invalid WANDB_API_KEY length: must be exactly 40 chars, got {len(wandb_api_key)}")
wandb.login(key=wandb_api_key)


# Import composite reward model (assumes rewards.py is in the same directory and modified to remove cosine/repetition)
from rewards import CompositeRewardModel, RewardConfig


# Global Constants
SYSTEM_PROMPT_1 = """
You are an expert multilingual medical doctor. When answering a medical question, follow these steps:\n1. First, search your internal knowledge base thoroughly for relevant background information about the topic.\n2. Understand and reason the question fully in English first.\n3. Reason mainly in English, but code-switch naturally into the target language whenever useful for clarity or domain accuracy.\n4. Consider multiple perspectives and potential answers before settling on your final response.\n5. Evaluate the confidence in your answer based on the information available to you.\n6. Provide the final answer clearly in the target language, making sure it's well-supported by your reasoning.\n7. If there are significant uncertainties or gaps in your knowledge, acknowledge them transparently.\n\nYour goal is to provide accurate, well-reasoned responses that demonstrate depth of understanding, not just surface-level answers."
""".strip()

SYSTEM_PROMPT_2 = """
You are an expert multilingual medical doctor. When answering a medical question, think and reason mainly in English with natural code-switching to the target language. Use multi-step reasoning wrapped in <step> tags inside <thinking> tags and the final direct answer in the language of the question inside <answer> </answer> tags.
""".strip()

RANDOM_SEED = 42
SFT_MODEL_PATH = "........."


STAGE_DATASET_PATH = "/standard/AikyamLab/eric/Multilingual medical reasoning/conversational finetuning/NEW_ALGORITHM_3RD/QWEN2.5_1.5B-INSTRUCT/Curriculum_RFT/staged/high"



# ============================================================================
# DATASET CLASS
# ============================================================================
class MedicalDataset(TorchDataset):
    """Custom dataset for loading multilingual medical JSONL files"""
    def __init__(self, data_path: str):
        super().__init__()
        self.samples = []
        jsonl_files = [f for f in os.listdir(data_path) if f.endswith('.jsonl')]
        for file in jsonl_files:
            file_path = os.path.join(data_path, file)
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            sample = json.loads(line.strip())
                            self.samples.append(sample)
                        except json.JSONDecodeError:
                            logging.warning(f"Skipping invalid JSON line in {file}")
        random.seed(RANDOM_SEED)
        random.shuffle(self.samples)
        logging.info(f"Loaded {len(self.samples)} samples from {data_path}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        language = sample['language']
        question = sample['question']
        # Format prompt as in SFT (only question is passed to model for generation)
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_1},
            {"role": "system", "content": SYSTEM_PROMPT_2},
            {"role": "user", "content": f"The question is in {language}. {question} Please think carefully with English-guided reasoning and code-switching, return your reasoning inside <thinking> </thinking> tags, and the final direct answer inside <answer> </answer> tags. Final answer ONLY in the language of the question."},
        ]
        return {
            'prompt': prompt_messages,
            'question': sample['question'],
            'answer': sample['answer'],
            'reasoning': sample['reasoning'],
            'language': sample['language'],
        }

        
# ============================================================================
# REWARD FUNCTION FOR GRPO
# ============================================================================
def get_reward_funcs():
    """Define reward function using CompositeRewardModel (modified to exclude cosine and repetition)"""
    reward_model = CompositeRewardModel(RewardConfig())
    def composite_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        # Retrieve dataset columns from kwargs
        questions = kwargs.get('question', [])
        answers = kwargs.get('answer', [])
        reasonings = kwargs.get('reasoning', [])
        languages = kwargs.get('language', [])
        # Reconstruct samples list (for reward computation against ground truth reasoning + answer)
        samples = []
        for q, a, r, l in zip(questions, answers, reasonings, languages):
            samples.append({
                'question': q,
                'answer': a,
                'reasoning': r,
                'language': l,
            })

            
        # Add printing logic for first few batches/prompts (for verification)
        global print_counter
        if 'print_counter' not in globals():
            print_counter = 0
        if print_counter < 2:  # Print for first 2 batches (adjust as needed)
            for idx in range(len(samples)):
                if idx >= 2: break  # Limit to first 2 prompts per batch
                print(f"\n=== Prompt {print_counter * len(samples) + idx + 1} ===")
                print(f"Question: {samples[idx]['question']}")
                print(f"Ground Truth Reasoning: {samples[idx]['reasoning']}")
                print(f"Ground Truth Answer: {samples[idx]['answer']}")
                
                for gen_idx, completion in enumerate(completions[idx]):
                    # Flexible extraction (as already in your code)
                    if isinstance(completion, dict):
                        generated = completion.get('content', '')
                    elif isinstance(completion, list) and completion and isinstance(completion[0], dict):
                        generated = completion[0].get('content', '')
                    else:
                        generated = str(completion)
                    
                    print(f"Generation {gen_idx + 1}: {generated}")
                    
                    # Compute and print reward for this generation
                    reward_dict = reward_model.compute_single_reward(samples[idx], generated)
                    print(f"Reward for Generation {gen_idx + 1}: {reward_dict['total_reward']:.3f}")
                    print(f"Components: {reward_dict['components']}")
            
            print_counter += 1
        rewards = []
        for comp_list, sample in zip(completions, samples):
            group_rewards = []
            for completion in comp_list:
                # Flexible extraction to handle dict, list of dicts, or str formats from generation
                if isinstance(completion, dict):
                    generated = completion.get('content', '') # Direct dict case
                elif isinstance(completion, list) and completion and isinstance(completion[0], dict):
                    generated = completion[0].get('content', '') # List of dicts
                else:
                    generated = str(completion) # Fallback to string
                if not generated:
                    group_rewards.append(0.0) # Default for invalid/empty generation
                    continue
                # Compute reward using ground truth (reasoning + answer) and generated response
                reward_dict = reward_model.compute_single_reward(sample, generated)
                group_rewards.append(reward_dict['total_reward'])
            # Aggregate: mean reward over generations for the prompt
            agg_reward = sum(group_rewards) / len(group_rewards) if group_rewards else 0.0
            rewards.append(agg_reward)
        return rewards
    return [composite_reward_func]

    
# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
@dataclass
class TrainingConfig:
    sft_model_path: str = field(default=SFT_MODEL_PATH)
    dataset_path: str = field(default=STAGE_DATASET_PATH)
    wandb_project: Optional[str] = field(default="RFT_code_switch_qwen1-5B_stage_one")
    block_size: int = field(default=32768)
    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project

        
# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train():
    # Parse arguments
    parser = transformers.HfArgumentParser((TrainingConfig, GRPOConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    # Load SFT model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_path, use_fast=True)
    # Set up tokenizer for Qwen
    if "Qwen" in config.sft_model_path or "qwen" in config.sft_model_path:
        tokenizer.pad_token = "<|fim_pad|>"
    # Load dataset
    dataset = MedicalDataset(config.dataset_path)
    # Prepare GRPO-specific args
    args.max_prompt_length = config.block_size // 2
    args.max_completion_length = config.block_size // 2
    args.eval_strategy = "no" # No evaluation during training
    # Initialize GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=None, # No eval
        processing_class=tokenizer,
        reward_funcs=get_reward_funcs(),
    )

    
    # Custom callback to pass samples to reward func via kwargs
    class SamplePassingCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            logging.info("Starting GRPO training")
    trainer.add_callback(SamplePassingCallback())
    # Train and save
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()
if __name__ == "__main__":
    train()