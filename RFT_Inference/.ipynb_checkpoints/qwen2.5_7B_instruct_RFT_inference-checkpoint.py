
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
from datasets import load_dataset
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------
# Define Constants and Configurations
# -------------------------------
# List of supported languages for processing 
    "amharic", "bengali", "french", "hausa", "hindi", "japanese",
    "korean", "spanish", "swahili", "thai", "turkish", "vietnamese", "yoruba"
]

# Path to the test dataset directory containing language-specific JSONL files.
test_dir = Path("/standard/AikyamLab/eric/Multilingual medical reasoning/conversational finetuning/NEW_ALGORITHM/DATASET/test_dataset/json_outputs")

# Path to the fine-tuned model checkpoint.
model_name = "model_here"

# System prompts used during fine-tuning (ensures responses are in the query's language with reasoning).
system_prompt_1 = "You are an expert multilingual medical doctor. When answering a medical question, follow these steps:\n1. First, search your internal knowledge base thoroughly for relevant background information about the topic.\n2. Understand and reason the question fully in English first.\n3. Reason mainly in English, but code-switch naturally into the target language whenever useful for clarity or domain accuracy.\n4. Consider multiple perspectives and potential answers before settling on your final response.\n5. Evaluate the confidence in your answer based on the information available to you.\n6. Provide the final answer clearly in the target language, making sure it's well-supported by your reasoning.\n7. If there are significant uncertainties or gaps in your knowledge, acknowledge them transparently.\n\nYour goal is to provide accurate, well-reasoned responses that demonstrate depth of understanding, not just surface-level answers."

system_prompt_2 = "You are an expert multilingual medical doctor. When answering a medical question, think and reason mainly in English with natural code-switching to the target language. Use multi-step reasoning wrapped in <step> tags inside <thinking> tags and the final direct answer in the language of the question inside <answer> </answer> tags."

# -------------------------------
# Load Model and Tokenizer
# -------------------------------
# Load the causal language model from the fine-tuned checkpoint in full precision.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically distribute across available devices (e.g., GPU).
    dtype=torch.bfloat16  # Data type for model weights (aligns with fine-tuning for efficiency).
)

# Load the tokenizer from the same checkpoint.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# -------------------------------
# Response Generation Function
# -------------------------------
# This function generates a response for a given question using the model's chat template.
# It formats the input as a conversation (system + user), generates output, and decodes it.
def generate_response(question, language, max_new_tokens=1024):
    # Prepare the message list for the chat template, matching fine-tuning format.
    messages = [
        {"role": "system", "content": system_prompt_1},
        {"role": "system", "content": system_prompt_2},
        {"role": "user", "content": f"The question is in {language}. {question} Please think carefully with English-guided reasoning and code-switching, return your reasoning inside <thinking> </thinking> tags, and the final direct answer inside <answer> </answer> tags. Final answer ONLY in the language of the question."},
    ]
    
    # Apply the Qwen2.5 chat template to format the prompt correctly (adds role tokens like <im_start>).
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Adds a generation prompt token for instruct models.
    )
    
    # Tokenize the prompt and move to the model's device.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate the response with no-gradient context to save memory.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Limit new tokens generated (controls response length).
            do_sample=True,  # Enable sampling for varied but coherent responses.
            temperature=0.6,  # Controls randomness (lower = more deterministic).
            top_p=0.95,  # Nucleus sampling threshold.
            top_k=20,  # Top-K sampling limit.
            repetition_penalty=1.2,  # Penalize repetition for more natural output.
            eos_token_id=tokenizer.eos_token_id,  # Stop at end-of-sequence token.
        )
    
    # Decode the generated tokens, skipping the input prompt and special tokens.
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()  # Clean up whitespace.


# -------------------------------
# Main Execution Block
# -------------------------------
# This block handles user input, processes each language's dataset, generates responses row-by-row,
# and saves results to CSV with resuming support and ability to extend samples.
if __name__ == "__main__":
    # Get user input for language selection and sample count.
    selected_language = input("Enter the language to process (or 'all' for all languages): ").strip().lower()
    num_samples = int(input("Enter the number of samples per language (-1 for all): "))
    
    # Determine languages to process.
    if selected_language == 'all':
        process_languages = languages
    else:
        process_languages = [selected_language]
    
    # Create output directory if it doesn't exist.
    output_dir = "finetuned_qwen2.5_7B_Instruct_inference_testdataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each selected language.
    for lang in process_languages:
        if lang not in languages:
            print(f"Invalid language: {lang}")
            continue
        
        # Load the language-specific JSONL file using datasets library.
        jsonl_path = test_dir / f"{lang}_sft.jsonl"
        if not jsonl_path.exists():
            print(f"File not found: {jsonl_path}")
            continue
        
        dataset = load_dataset('json', data_files=str(jsonl_path), split='train')
        df = dataset.to_pandas()[['question', 'answer']]
        
        # Define output CSV filename and path.
        new_filename = f"{lang}_finetuned_qwen2.5_7B_instruct_inference_data.csv"
        save_path = os.path.join(output_dir, new_filename)
        
        # If existing CSV exists, load it and transfer 'Test Response' to the full df.
        completed_rows = 0
        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path)
            if 'Test Response' in existing_df.columns:
                for idx in range(min(len(existing_df), len(df))):
                    if pd.notna(existing_df.at[idx, 'Test Response']):
                        df.at[idx, 'Test Response'] = existing_df.at[idx, 'Test Response']
                completed_rows = existing_df[existing_df['Test Response'].notna()].shape[0]
                print(f"Resuming from {completed_rows} completed rows in {new_filename}")
        
        # Ensure 'Test Response' column exists.
        if 'Test Response' not in df.columns:
            df['Test Response'] = ''
        
        # Limit to the number of samples if specified.
        total_rows = len(df)
        process_rows = total_rows if num_samples == -1 else min(num_samples, total_rows)
        df = df.iloc[:process_rows].copy()
        
        # Generate responses for each row, skipping completed ones, and save progressively.
        for idx in range(len(df)):
            if pd.notna(df.at[idx, 'Test Response']) and df.at[idx, 'Test Response'].strip() != '':
                continue
            
            question = df.at[idx, 'question']
            print(f"Generating response for row {idx+1}/{process_rows} in {lang}...")
            test_response = generate_response(question, lang)
            df.at[idx, 'Test Response'] = test_response
            
            # Save the DataFrame after each generation to avoid data loss (only selected columns).
            df[['question', 'answer', 'Test Response']].to_csv(save_path, index=False)
        
        print(f"Completed and saved: {save_path}")