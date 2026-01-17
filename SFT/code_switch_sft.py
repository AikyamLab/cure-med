import os
import sys
sys.path.append('..')
sys.path.append('.')
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, DatasetDict, disable_caching
disable_caching()
import transformers
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from pathlib import Path
import torch

@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.
    """
    model_name: str = field(default="Qwen/Qwen2.5-1.5B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="language_switch_qwen1.5B_sft")
    data_dir: Optional[str] = field(default='/standard/AikyamLab/eric/Multilingual medical reasoning/conversational finetuning/NEW_ALGORITHM_3RD/Dataset/SFT_data')
    
    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project

def prepare_dataset(dataset, tokenizer):
    """
    Prepare the dataset by formatting each example into a conversational structure,
    tokenizing it using the default chat template, and masking labels to compute loss
    only on the assistant's response.
    """
    def _format_and_tokenize(examples):
        questions = examples['question']
        reasonings = examples['reasoning']
        answers = examples['answer']
        languages = examples['language']
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        
        response_template = "<|im_start|>assistant\n"
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        
        for question, reasoning, answer, language in zip(questions, reasonings, answers, languages):
            messages = [
                {"role": "system", "content": "You are an expert multilingual medical doctor. When answering a medical question, follow these steps:\n1. First, search your internal knowledge base thoroughly for relevant background information about the topic.\n2. Understand and reason the question fully in English first.\n3. Reason mainly in English, but code-switch naturally into the target language whenever useful for clarity or domain accuracy.\n4. Consider multiple perspectives and potential answers before settling on your final response.\n5. Evaluate the confidence in your answer based on the information available to you.\n6. Provide the final answer clearly in the target language, making sure it's well-supported by your reasoning.\n7. If there are significant uncertainties or gaps in your knowledge, acknowledge them transparently.\n\nYour goal is to provide accurate, well-reasoned responses that demonstrate depth of understanding, not just surface-level answers."},
                {"role": "system", "content": "You are an expert multilingual medical doctor. When answering a medical question, think and reason mainly in English with natural code-switching to the target language. Use multi-step reasoning wrapped in <step> tags inside <thinking>."},
                {"role": "user", "content": f"The question is in {language}. {question} Please think carefully with English-guided reasoning and code-switching, return your reasoning inside <thinking> </thinking> tags, and the final direct answer inside <answer> </answer> tags. Final answer ONLY in {language}."},
                {"role": "assistant", "content": f"{reasoning}\n{answer}"}
            ]
            
            # Tokenize using default chat template
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            
            input_ids = tokenized[0]  # Since return_tensors="pt", it's a tensor
            attention_mask = torch.ones_like(input_ids)  # Assume all attended, or use if provided
            
            # Create labels and mask prompt (loss only on assistant response)
            labels = input_ids.clone()
            # Find the starting position of the assistant response content
            template_len = len(response_template_ids)
            found = False
            for i in range(len(input_ids) - template_len + 1):
                if torch.all(input_ids[i:i+template_len] == torch.tensor(response_template_ids)):
                    # Mask everything before the content after the template
                    start = i + template_len
                    labels[:start] = -100
                    found = True
                    break
            if not found:
                logging.warning("Response template not found in tokenized input.")
                labels[:] = -100  
            
            input_ids_list.append(input_ids.tolist())
            labels_list.append(labels.tolist())
            attention_mask_list.append(attention_mask.tolist())
        
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list
        }
    
    dataset = dataset.map(_format_and_tokenize, batched=True, remove_columns=dataset.column_names, num_proc=1)
    return DatasetDict({'train': dataset})

class EmptyCacheCallback(TrainerCallback):
    """
    Custom callback to empty CUDA cache at the end of each training step, with sync.
    """
    def on_step_end(self, args, state, control, **kwargs):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # Sync all ranks before flushing
        torch.cuda.empty_cache()

def train():
    # Parsing command-line arguments
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    
    # Loading model and tokenizer (uses default chat template)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency on larger models
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    
    # Set pad token for Qwen models
    if "Qwen" in config.model_name or "qwen" in config.model_name:
        tokenizer.pad_token = "<|fim_pad|>"
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Loading and preparing dataset (pre-tokenized with masked labels)
    data_files = [str(file) for file in Path(config.data_dir).glob("*.jsonl")]
    hf_dataset = load_dataset('json', data_files=data_files, split='train')
    hf_dataset = hf_dataset.filter(
        lambda example: all(key in example and example[key] is not None for key in ['question', 'reasoning', 'answer', 'language']),
        num_proc=1
    )
    hf_dataset = hf_dataset.shuffle(seed=42)
    dataset = prepare_dataset(hf_dataset, tokenizer)['train']
    print(dataset)
    
    # Setting up trainer (no custom collator needed, as preprocessed)
    args.max_seq_length = config.block_size
    args.gradient_checkpointing = True  # Enable for memory efficiency on large models
    args.bf16 = True  # Use bf16 mixed precision if supported
    trainer = trl.SFTTrainer(
        model=model,
        processing_class=tokenizer,  # Use processing_class for compatibility
        train_dataset=dataset,
        args=args,
        callbacks=[EmptyCacheCallback()],  # Add callback to empty cache
    )
    
    # Run training and save
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()

if __name__ == "__main__":
    train()

