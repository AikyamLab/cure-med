#!/usr/bin/env python3
"""
GRPO Training Script for Multilingual Medical Reasoning
Using Qwen2.5-7B-Instruct SFT Model with Composite Reward Function
Inspired by Curr-ReFT for Curriculum-Based Reinforcement Fine-Tuning
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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainerCallback
import transformers
from trl import GRPOTrainer, GRPOConfig
import wandb

# WandB setup (hardcoded API key - note: hardcoding is insecure; use for testing only)
wandb_api_key = "07bd5401faabf624853f37acef31e87a3d7ef582"
if len(wandb_api_key) != 40:
    raise ValueError(f"Invalid WANDB_API_KEY length: must be exactly 40 chars, got {len(wandb_api_key)}")
wandb.login(key=wandb_api_key)

# Import composite reward model
from rewards import CompositeRewardModel, RewardConfig

# Global Constants
SYSTEM_PROMPT_1 = """
You are an expert multilingual medical doctor. When answering a medical question, follow these steps:
1. First, search your internal knowledge base thoroughly for relevant background information about the topic.
2. Think and reason carefully in the same language as the question (for example, if the question is in Hindi, then think and reason in Hindi).
3. Consider multiple perspectives and potential answers before settling on your final response.
4. Evaluate the confidence in your answer based on the information available to you.
5. Provide the final answer clearly in the same language as the question, making sure it's well-supported by your reasoning.
6. If there are significant uncertainties or gaps in your knowledge, acknowledge them transparently.
Your goal is to provide accurate, well-reasoned responses that demonstrate depth of understanding, not just surface-level answers.
""".strip()
SYSTEM_PROMPT_2 = """
You are an expert multilingual medical doctor. When answering a medical question, think and reason ONLY in the same language as the question (for example, if question is in Hindi then think and reason in Hindi). Use multi-step reasoning wrapped in <step> tags inside <think>.
""".strip()

RANDOM_SEED = 42
SFT_MODEL_PATH = "/standard/AikyamLab/eric/Multilingual medical reasoning/conversational finetuning/NEW_ALGORITHM/RFT_WITH_GRPO/Training_Stages/Stage_one_training/stage_one_high_output_20251011_001231"

STAGE_DATASET_PATH = "/standard/AikyamLab/eric/Multilingual medical reasoning/conversational finetuning/NEW_ALGORITHM/RFT_WITH_GRPO/Dataset_stages/stage_two/high"

NUM_GENERATIONS = 8  # GRPO: 8 responses per prompt

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
        # Format prompt as in SFT
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_1},
            {"role": "system", "content": SYSTEM_PROMPT_2},
            {"role": "user", "content": f"The question is in {sample['language']}. {sample['question']} Please think carefully and return your reasoning inside <think> </think> tags, and the final direct answer inside <answer> </answer> tags. Respond ONLY in the language of the question."},
        ]
        return {
            'prompt': prompt_messages,
            'question': sample['question'],
            'answer': sample['answer'],
            'reasoning': sample['reasoning'],
            'language': sample['language'],
            'question_type': sample.get('question_type', 'open'),  # Default to open if not specified
        }

# ============================================================================
# REWARD FUNCTION FOR GRPO
# ============================================================================
def get_reward_funcs():
    """Define reward function using CompositeRewardModel"""
    reward_model = CompositeRewardModel(RewardConfig())

    def composite_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        # Retrieve dataset columns from kwargs (passed by trainer)
        questions = kwargs.get('question', [])
        answers = kwargs.get('answer', [])
        reasonings = kwargs.get('reasoning', [])
        languages = kwargs.get('language', [])
        question_types = kwargs.get('question_type', [])

        # Reconstruct samples list
        samples = []
        for q, a, r, l, qt in zip(questions, answers, reasonings, languages, question_types):
            samples.append({
                'question': q,
                'answer': a,
                'reasoning': r,
                'language': l,
                'question_type': qt
            })

        # Add printing logic for first few batches/prompts
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
                    # Flexible extraction to handle dict, list of dicts, or str formats from generation
                    if isinstance(completion, dict):
                        generated = completion.get('content', '')  # Direct dict case
                    elif isinstance(completion, list) and completion and isinstance(completion[0], dict):
                        generated = completion[0].get('content', '')  # List of dicts
                    else:
                        generated = str(completion)  # Fallback to string
                    
                    print(f"Generation {gen_idx + 1}: {generated}")
                    
                    # Compute and print reward for this generation
                    reward_dict = reward_model.compute_single_reward(samples[idx], generated)
                    print(f"Reward for Generation {gen_idx + 1}: {reward_dict['total_reward']}")
                    print(f"Components: {reward_dict['components']}")
            
            print_counter += 1

        rewards = []
        for comp_list, sample in zip(completions, samples):
            group_rewards = []
            for completion in comp_list:
                # Flexible extraction to handle dict, list of dicts, or str formats from generation
                if isinstance(completion, dict):
                    generated = completion.get('content', '')  # Direct dict case
                elif isinstance(completion, list) and completion and isinstance(completion[0], dict):
                    generated = completion[0].get('content', '')  # List of dicts
                else:
                    generated = str(completion)  # Fallback to string
                
                if not generated:
                    group_rewards.append(0.0)  # Default for invalid/empty generation
                    continue
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
    wandb_project: Optional[str] = field(default="stage_two_high_grpo_training")
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
    args.eval_strategy = "no"  # No evaluation during training

    # Initialize GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=None,  # No eval
        processing_class=tokenizer,
        reward_funcs=get_reward_funcs(),
    )

    # Custom callback to pass samples to reward func via kwargs
    class SamplePassingCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            pass  # No eval

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
