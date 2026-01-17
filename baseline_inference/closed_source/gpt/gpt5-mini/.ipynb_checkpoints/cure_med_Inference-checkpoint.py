# -------------------------------
# GPT-5-mini Batch Inference Script for Test Datasets
# -------------------------------
import os
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI


API_KEY = "......"
client = OpenAI(api_key=API_KEY)

# List of languages
languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese",
    "Korean", "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]

# Dataset path
test_dir = Path("/standard/AikyamLab/eric/Multilingual medical reasoning/Unsloth_Finetune_2/test_dataset")

# Model name
model_name = "gpt-5-mini-2025-08-07"  
# System prompt
system_prompt = (
    "You are an expert multilingual medical doctor. Please reason step by step about this medical query, "
    "then provide the correct answer exclusively in the same language as the question—do not use English at all "
    "unless the question itself is in English. Your entire response must be in the language of the question."
)

def _extract_text_from_response(resp) -> str:
    """
    Prefer SDK convenience property `output_text` where available; fall back to parsing output items.
    """
    try:
        ot = getattr(resp, "output_text", None)
        if isinstance(ot, str) and ot.strip():
            return ot.strip()
    except Exception:
        pass

    chunks = []

    output = getattr(resp, "output", None)
    if not output:
        return ""

    for item in output:
        item_type = getattr(item, "type", None)
        if item_type != "message":
            continue

        content_list = getattr(item, "content", None) or []
        for c in content_list:
            c_type = getattr(c, "type", None)
            if c_type in ("output_text", "text"):
                txt = getattr(c, "text", None)
                if isinstance(txt, str) and txt:
                    chunks.append(txt)

    return "".join(chunks).strip()

# Function to generate response
def generate_response(question, max_new_tokens=1024, max_retries=5):
    """
    Correct GPT-5 mini usage with the Responses API:
      - Put system content in `instructions` (not as a system role message).
      - Do NOT pass `frequency_penalty` (unsupported in Responses API).
      - Do NOT pass `temperature` / `top_p` for older GPT-5 family snapshots (can raise errors).
      - `reasoning.effort="none"` is not supported for gpt-5-mini; use minimal/low/medium/high instead.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model_name,
                input=question,                 
                instructions=system_prompt,     
                reasoning={"effort": "minimal"},
                max_output_tokens=max_new_tokens
            )
            return _extract_text_from_response(resp)
        except Exception as e:
            last_err = e
            # backoff for transient failures (rate limits, network hiccups)
            sleep_s = min(2 ** (attempt - 1), 16)
            print(f"[WARN] API call failed (attempt {attempt}/{max_retries}): {e}")
            time.sleep(sleep_s)

    print(f"[ERROR] Exhausted retries. Last error: {last_err}")
    return ""  # leave blank so your resume logic can try again later

if __name__ == "__main__":
    # User input for scalability
    selected_language = input("Enter the language to process (or 'all' for all languages): ").strip().capitalize()
    num_samples = int(input("Enter the number of samples per language (-1 for all): "))

    if selected_language.lower() == 'all':
        process_languages = languages
    else:
        process_languages = [selected_language]

    # Output directory
    output_dir = "gpt_5_mini_inference_testdataset"
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

        new_filename = f"{lang}_gpt_5_mini_inference_data.csv"
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
            test_response = generate_response(question)
            df.at[idx, 'Test Response'] = test_response

            # Save after each generation
            df.to_csv(save_path, index=False)

        print(f"Completed and saved: {save_path}")
