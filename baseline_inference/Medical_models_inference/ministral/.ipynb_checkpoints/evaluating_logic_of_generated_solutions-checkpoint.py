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
parser = argparse.ArgumentParser(description="Evaluate logical reasoning of responses using gpt-4o-mini as a judge.")
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
dataset_dir = "/standard/AikyamLab/eric/Multilingual medical reasoning/Baselines/other_models_inference/ministral/Dataset"
output_dir = os.path.join(os.getcwd(), "BaseModel_logic_evaluation")
os.makedirs(output_dir, exist_ok=True)

def get_logic_evaluation(question, response, test_response):
    """
    Evaluate the logic of the model-generated response using GPT.
    Returns "True" if correct, "False" otherwise.
    """
    user_prompt = f"""
<Question>
{question}
</Question>
<Model Response>
{response}
</Model Response>
<Test Response>
{test_response}
</Test Response>
You are provided with a medical question, a model-generated response (which may include reasoning), and a test response (the final answer). 
Determine if the model's reasoning is logically sound, consistent, and leads to a medically accurate conclusion. Verify if the test response correctly answers the question based on standard medical knowledge. 
Your task is to simply output "True" if the criteria are met (logical and accurate), and "False" otherwise.
"""
    for attempt in range(3):
        try:
            gen_response = client.chat.completions.create(
                model="gpt-4o-mini",
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
            print(f"Attempt {attempt+1} failed for logic evaluation. Error: {e}")
            time.sleep(2)
    print("Skipping logic evaluation after failures.")
    return None

# Process each selected language
for lang in selected_languages:
    input_csv_path = os.path.join(dataset_dir, f"test_result_{lang}_Ministral-8B.csv")
    output_csv_path = os.path.join(output_dir, f"{lang}_ministral-8b_logic_evaluation.csv")
    
    print(f"Loading data for {lang} from {input_csv_path}...")
    
    # Load the full input CSV
    df = pd.read_csv(input_csv_path)
    
    # Ensure evaluation column exists
    if 'Logic_Score' not in df.columns:
        df['Logic_Score'] = None
    
    # If output CSV exists, merge existing evaluations
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        # Update corresponding rows in df with existing values
        min_len = min(len(df), len(existing_df))
        df.loc[:min_len-1, 'Logic_Score'] = existing_df.loc[:min_len-1, 'Logic_Score']
        print(f"Resuming from existing output CSV for {lang}. Evaluated rows will be skipped.")
    else:
        print(f"Starting fresh for {lang}.")
    
    # Apply num_samples limit if specified
    if args.num_samples != -1:
        df = df.head(args.num_samples)
        print(f"Limiting to first {args.num_samples} samples for {lang}.")
    
    # Find unevaluated rows
    unevaluated_mask = df['Logic_Score'].isnull()
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
        response = row.get('Response', '')
        test_response = row.get('Test_Response', '')
        
        if not all(isinstance(x, str) for x in [question, response, test_response]):
            print(f"Skipping invalid row at index {idx} for {lang}")
            continue
        
        logic_evaluation = get_logic_evaluation(question, response, test_response)
        if logic_evaluation is not None:
            df.at[idx, 'Logic_Score'] = logic_evaluation
        
        # Save incrementally after each evaluation
        df.to_csv(output_csv_path, index=False)
        time.sleep(2)
    
    print(f"\n:white_check_mark: Completed logic evaluation for {lang}! Results saved to: {output_csv_path}")