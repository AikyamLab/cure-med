##This script is for generating responses in different languages from the llama model 

from huggingface_hub import login
login(token="your key here")


import os

os.environ['HF_HOME'] = '/scratch/reh6ed/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/reh6ed/hf_cache'

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from pathlib import Path



# Configuration: List of supported languages from your dataset
languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese",
    "Korean", "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]

# Dataset path: Adjust if your test directory is different
test_dir = Path("your path")

# Model and Tokenizer Loading
# Load the quantized model and tokenizer for Llama-3.1-8B-Instruct
model_name = "meta-llama/Llama-3.3-70B-Instruct" ###change to the llama model you are using

# Quantization config: 4-bit for memory efficiency, with float16 compute to match inputs and avoid dtype warnings
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16  # Aligns with input dtype for faster, warning-free inference
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically distributes across available GPUs
    
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.eos_token_id  # Prevents "Setting pad_token_id" messages

# System Prompt: Defines the model's role for consistent medical reasoning in the query's language
system_prompt = (
    "You are an expert multilingual medical doctor. "
    "Please provide the correct answer to this medical query in the given language only after proper reasoning."
)

# Core Function: Generate a reasoned response for a given question
def generate_response(question, max_new_tokens=1024):
    """
    Generates a response using the chat template and model generation.
    
    Args:
        question (str): The medical query in the target language.
        max_new_tokens (int): Maximum tokens to generate (default: 1024 for detailed reasoning).
    
    Returns:
        str: The generated response, stripped of special tokens.
    """
    # Format messages with system and user roles
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    # Apply Llama's chat template to create the prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Adds the assistant prompt token
    )
    
    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response with no_grad for efficiency
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # Enables sampling for varied, coherent responses
            temperature=0.6,  # Balanced creativity (lower for more deterministic)
            top_p=0.95,  # Nucleus sampling for diversity
            top_k=20,  # Limits to top 20 tokens for focus
            repetition_penalty=1.2,  # Discourages repetitive phrases
            eos_token_id=tokenizer.eos_token_id,  # Ensures proper stopping
        )
    
    # Decode only the new tokens (post-input)
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):], 
        skip_special_tokens=True
    )
    return response.strip()

# Main Execution Block
if __name__ == "__main__":
    # User Inputs: For flexible processing
    selected_language = input("Enter the language to process (or 'all' for all languages): ").strip().capitalize()
    num_samples = int(input("Enter the number of samples per language (-1 for all): "))
    
    # Determine languages to process
    if selected_language.lower() == 'all':
        process_languages = languages
    else:
        process_languages = [selected_language]
    
    # Output Directory: Organized by model for easy tracking
    output_dir = "llama3.1_70B_Instruct_inference_testdataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process Each Language
    for lang in process_languages:
        # Validation: Skip invalid languages
        if lang not in languages:
            print(f"Invalid language: {lang}")
            continue
        
        # Load Dataset
        csv_path = test_dir / f"{lang}.csv"
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        process_rows = total_rows if num_samples == -1 else min(num_samples, total_rows)
        df = df.iloc[:process_rows].copy()
        df['Test Response'] = ''  # Initialize response column
        
        # Output File Setup
        new_filename = f"{lang}_llama3.1_8B_instruct_inference_data.csv"
        save_path = os.path.join(output_dir, new_filename)
        
        # Resume Logic: Load existing progress if available
        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path)
            if 'Test Response' in existing_df.columns:
                completed_rows = existing_df[existing_df['Test Response'].notna()].shape[0]
                if completed_rows > 0:
                    print(f"Resuming from {completed_rows} completed rows in {new_filename}")
                    # Copy completed responses back to df
                    for i in range(min(completed_rows, len(df))):
                        df.at[i, 'Test Response'] = existing_df.at[i, 'Test Response']
        
        # Generate Responses: Row-by-row with incremental saving
        for idx in range(len(df)):
            # Skip if already completed
            if df.at[idx, 'Test Response']:
                continue
            
            question = df.at[idx, 'Question']
            print(f"Generating response for row {idx+1}/{process_rows} in {lang}...")
            
            # Generate and assign response
            test_response = generate_response(question)
            df.at[idx, 'Test Response'] = test_response
            
            # Save progress after each response to disk
            df.to_csv(save_path, index=False)
        
        print(f"Completed and saved: {save_path}")