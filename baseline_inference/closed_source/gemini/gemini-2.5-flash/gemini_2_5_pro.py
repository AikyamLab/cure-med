# -------------------------------
# Gemini (DeepInfra) Batch Inference Script for Test Datasets
# (Keeps the same dataset path, languages, and resume/saving logic)
# -------------------------------
import os
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI

# -------------------------------
# DeepInfra API settings
# -------------------------------
# Hardcode your DeepInfra API key here (use the SAME key you tested)
DEEPINFRA_API_KEY = "......"

# DeepInfra provides an OpenAI-compatible base URL
client = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="........",
)

# Model name (DeepInfra model id)
# You tested: "google/gemini-2.5-pro"
# Cheaper alternative: "google/gemini-2.5-flash"
model_name = "google/gemini-2.5-flash"


# -------------------------------
# Your existing settings
# -------------------------------
languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese",
    "Korean", "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]

test_dir = Path("/standard/AikyamLab/eric/Multilingual medical reasoning/Unsloth_Finetune_2/test_dataset")

# System prompt (same intent as your GPT prompt, but explicitly prevents <think> leakage)
system_prompt = (
    "You are an expert multilingual medical doctor.\n"
    "Provide a concise clinical explanation and then the final answer.\n"
    "IMPORTANT: Do NOT output hidden thoughts, chain-of-thought, or <think>...</think>.\n"
    "Write exclusively in the same language as the question—do not use English unless the question is in English.\n"
    "Your entire response must be in the language of the question."
)

def _strip_think_blocks(text: str) -> str:
    """
    Some thinking models may include <think> blocks in the visible output.
    Remove them if present to keep outputs clean (like your GPT outputs).
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""

    if t.startswith("<think>"):
        end = t.find("</think>")
        if end != -1:
            return t[end + len("</think>"):].strip()
        # If no closing tag, drop the first paragraph as a fallback
        parts = t.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
        return ""

    return t

def generate_response(question, max_new_tokens=1024, max_retries=6):
    """
    DeepInfra OpenAI-compatible Chat Completions call for Gemini.


    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.6,
                top_p=0.95,
                max_tokens=max_new_tokens,
            )

            content = ""
            if resp and resp.choices and resp.choices[0].message:
                content = resp.choices[0].message.content or ""

            return _strip_think_blocks(content)

        except Exception as e:
            last_err = e
            sleep_s = min(2 ** (attempt - 1), 20)
            print(f"[WARN] Request failed (attempt {attempt}/{max_retries}): {e}")
            time.sleep(sleep_s)

    print(f"[ERROR] Exhausted retries. Last error: {last_err}")
    return ""  # leave blank so your resume logic can rerun later


if __name__ == "__main__":
    # User input for scalability (same structure as your GPT script)
    selected_language = input("Enter the language to process (or 'all' for all languages): ").strip().capitalize()
    num_samples = int(input("Enter the number of samples per language (-1 for all): "))

    if selected_language.lower() == 'all':
        process_languages = languages
    else:
        process_languages = [selected_language]

    # Output directory
    output_dir = "gemini_inference_testdataset"
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

        new_filename = f"{lang}_gemini_inference_data.csv"
        save_path = os.path.join(output_dir, new_filename)

        # Resume if file exists (kept the same logic)
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
