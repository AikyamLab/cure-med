import openai
from dotenv import load_dotenv
import pandas as pd
import argparse
import time
import os
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key with fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = "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"  # Fallback to provided key if env var not set

def setup_openai():
    """
    Configure the OpenAI API client.
    """
    if not OPENAI_API_KEY:
        raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY in your .env file")
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return client

# Initialize the OpenAI client
client = setup_openai()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate language of responses using gpt-4o-mini as a judge.")
parser.add_argument("--language", type=str, default=None, help="Language to evaluate (e.g., Amharic, Bengali, etc.). Required if --all is not used.")
parser.add_argument("--all", action="store_true", help="Process all languages instead of a single one.")
parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process per language. Use -1 to process all (default), or a positive number for a subset (e.g., 4 for testing).")
args = parser.parse_args()

# List of languages to process
languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese", "Korean",
    "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]

# Determine selected languages based on arguments
if args.all:
    selected_languages = languages
else:
    if not args.language:
        raise ValueError("Must provide --language or use --all flag.")
    selected_languages = [args.language]

# Define dataset and output directories
dataset_dir = "/standard/AikyamLab/eric/Multilingual medical reasoning/conversational finetuning/closed_source/gpt_4o_mini/gpt5-mini/gpt_5_mini_inference_testdataset"

output_dir = os.path.join(os.getcwd(), "BaseModel_language_evaluation")
os.makedirs(output_dir, exist_ok=True)

def get_language_evaluation(question, test_response, language):
    """
    Evaluate the language of the model-generated response using GPT.
    Returns "True" if criteria are met, "False" otherwise.
    """
    user_prompt = f"""
<Question>
{question}
</Question>
<Model Response>
{test_response}
</Model Response>
You are provided with a question in {language} and a model-generated response (<Model Response>).
Determine if the model response is entirely in {language} (the language of the question)? Check for no mixing with other languages, no unnecessary repetitions, no drifts, and a proper conclusion. Your task is to simply output "True" if the criteria is met, and "False" otherwise.
"""
    for attempt in range(3):
        try:
            gen_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": user_prompt}]
            )
            output = gen_response.choices[0].message.content.strip()
            if output.lower() == "true":
                return "True"
            elif output.lower() == "false":
                return "False"
            else:
                raise ValueError("Invalid output: not True or False")
        except Exception as e:
            print(f"Attempt {attempt+1} failed for language evaluation. Error: {e}")
            time.sleep(2)
    print("Skipping language evaluation after failures.")
    return None

# Process each selected language
for lang in selected_languages:
    input_csv_path = os.path.join(dataset_dir, f"{lang}_gpt_5_mini_inference_data.csv")
    output_csv_path = os.path.join(output_dir, f"{lang}_qwen2.5_14B_language_evaluation.csv")
    
    print(f"Loading data for {lang} from {input_csv_path}...")
    
    # Load the full input CSV
    df = pd.read_csv(input_csv_path)
    
    # Ensure evaluation column exists
    if 'Language_Score' not in df.columns:
        df['Language_Score'] = None
    
    # If output CSV exists, merge existing evaluations
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        # Update corresponding rows in df with existing values
        min_len = min(len(df), len(existing_df))
        df.loc[:min_len-1, 'Language_Score'] = existing_df.loc[:min_len-1, 'Language_Score']
        print(f"Resuming from existing output CSV for {lang}. Evaluated rows will be skipped.")
    else:
        print(f"Starting fresh for {lang}.")
    
    # Apply num_samples limit if specified
    if args.num_samples != -1:
        df = df.head(args.num_samples)
        print(f"Limiting to first {args.num_samples} samples for {lang}.")
    
    # Find unevaluated rows
    unevaluated_mask = df['Language_Score'].isnull()
    num_to_process = unevaluated_mask.sum()
    unevaluated_indices = df[unevaluated_mask].index.tolist()
    
    if num_to_process == 0:
        print(f"All rows already evaluated for {lang}. Skipping.")
        continue
    
    print(f"Evaluating {num_to_process} remaining responses in {lang}...")
    
    # Evaluate each unevaluated row
    for idx in tqdm(unevaluated_indices):
        row = df.loc[idx]
        question = row.get('Question', '')
        test_response = row.get('Test Response', '')
        
        if not all(isinstance(x, str) for x in [question, test_response]):
            print(f"Skipping invalid row at index {idx} for {lang}")
            continue
        
        language_evaluation = get_language_evaluation(question, test_response, lang)
        if language_evaluation is not None:
            df.at[idx, 'Language_Score'] = language_evaluation
        
        # Save incrementally after each evaluation
        df.to_csv(output_csv_path, index=False)
        time.sleep(1)
    
    print(f"\n:white_check_mark: Completed language evaluation for {lang}! Results saved to: {output_csv_path}")