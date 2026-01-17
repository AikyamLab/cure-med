# -------------------------------
# Apollo2-7B Batch Inference Script for Test Datasets
# -------------------------------
import os
os.environ['HF_HOME'] = '/scratch/reh6ed/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/reh6ed/hf_cache'


import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
import torch
from pathlib import Path

# List of languages
languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese",
    "Korean", "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]

# Dataset path
test_dir = Path("/standard/AikyamLab/eric/Multilingual medical reasoning/Unsloth_Finetune_2/test_dataset")

# Load the model and tokenizer with trust_remote_code=True
model_name = "FreedomIntelligence/Apollo2-7B"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16  # Set to float16 to match input dtype and avoid warning/slowdown
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically distributes across available GPUs (e.g., 2 GPUs)
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set pad token to eos token (common fix for models without a dedicated pad token)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Base system prompt template (dynamic per language)
system_prompt_template = "You are an expert multilingual medical doctor. The following query is in {lang}. Please reason step by step in {lang} only and provide the correct final answer to this medical query in {lang} only."

# Function to generate response
def generate_response(question, lang, max_new_tokens=1024):
    system_prompt = system_prompt_template.format(lang=lang)
    # Build prompt in the model's chat format
    prompt = f"User: {system_prompt}\n\n{question}\nAssistant:"
  
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Set generation config as per model recommendations
    generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.2
    )
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    # User input for scalability
    selected_language = input("Enter the language to process (or 'all' for all languages): ").strip().capitalize()
    num_samples = int(input("Enter the number of samples per language (-1 for all): "))
    if selected_language.lower() == 'all':
        process_languages = languages
    else:
        process_languages = [selected_language]
    # Output directory
    output_dir = "apollo2_7b_inference_testdataset"
    os.makedirs(output_dir, exist_ok=True)
    # Process each language
    for lang in process_languages:
        if lang not in languages:
            print(f"Invalid language: {lang}")
            continue
        csv_path = test_dir / f"{lang}.csv"
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        process_rows = total_rows if num_samples == -1 else min(num_samples, total_rows)
        df = df.iloc[:process_rows].copy()
        df['Test Response'] = ''  # Initialize column
        new_filename = f"{lang}_apollo2_7b_inference_data.csv"
        save_path = os.path.join(output_dir, new_filename)
        # Resume if file exists
        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path)
            if 'Test Response' in existing_df.columns:
                completed_rows = existing_df[existing_df['Test Response'].notna()].shape[0]
                if completed_rows > 0:
                    print(f"Resuming from {completed_rows} completed rows in {new_filename}")
                    df.iloc[:completed_rows] = existing_df.iloc[:completed_rows]
        # Generate responses row by row with progressive saving
        for idx in range(len(df)):
            if df.at[idx, 'Test Response']:  # Skip completed
                continue
            question = df.at[idx, 'Question']
            print(f"Generating response for row {idx+1}/{process_rows} in {lang}...")
            test_response = generate_response(question, lang)
            df.at[idx, 'Test Response'] = test_response
            # Save after each generation
            df.to_csv(save_path, index=False)
        print(f"Completed and saved: {save_path}")