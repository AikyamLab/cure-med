# -------------------------------
# Claude 3 Haiku Batch Inference Script for Test Datasets
# (Same dataset path, languages, and resume/saving logic)
# -------------------------------
import os
import time
import json
import pandas as pd
from pathlib import Path


ANTHROPIC_API_KEY = ".."

# Model name (Claude 3 Haiku)
MODEL_NAME = "claude-3-haiku-20240307"

# List of languages
languages = [
    "Amharic", "Bengali", "French", "Hausa", "Hindi", "Japanese",
    "Korean", "Spanish", "Swahili", "Thai", "Turkish", "Vietnamese", "Yoruba"
]

# Dataset path (same as yours)
test_dir = Path("/standard/AikyamLab/eric/Multilingual medical reasoning/Unsloth_Finetune_2/test_dataset")

# System prompt (Claude Messages API uses top-level `system`, not a "system" role message)
system_prompt = (
    "You are an expert multilingual medical doctor. "
    "Please reason step by step about this medical query, then provide the correct answer "
    "exclusively in the same language as the question—do not use English at all unless the question itself is in English. "
    "Your entire response must be in the language of the question."
)

# Generation controls
TEMPERATURE = 0.6
TOP_P = 0.95

def _extract_text_from_anthropic_response_dict(data: dict) -> str:
    """
    Anthropic /v1/messages returns content blocks like:
      { "content": [ { "type": "text", "text": "..." }, ... ] }
    """
    if not isinstance(data, dict):
        return ""
    blocks = data.get("content", [])
    out = []
    if isinstance(blocks, list):
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                txt = b.get("text", "")
                if isinstance(txt, str) and txt:
                    out.append(txt)
    return "".join(out).strip()

def _call_with_anthropic_sdk(question: str, max_tokens: int) -> str:
    """
   code block
    """
    from anthropic import Anthropic  # requires `pip install anthropic`

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    msg = client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        system=system_prompt,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        messages=[
            {"role": "user", "content": question}
        ],
    )

    # SDK returns Message with content blocks
    parts = []
    if hasattr(msg, "content") and isinstance(msg.content, list):
        for block in msg.content:
            # text blocks usually have `.text`
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts).strip()

def _call_with_requests(question: str, max_tokens: int) -> str:
    """
    Raw HTTP fallback to Anthropic Messages API:
      POST https://api.anthropic.com/v1/messages
      headers: x-api-key, anthropic-version, content-type
    """
    import requests

    url = "..........."
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "messages": [{"role": "user", "content": question}],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Anthropic HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    return _extract_text_from_anthropic_response_dict(data)

def generate_response(question: str, max_new_tokens: int = 1024, max_retries: int = 6) -> str:
    """
    Robust generation with retries (handles transient 429/5xx/529 style issues).
    Tries Anthropic SDK first; falls back to raw HTTP if SDK isn't installed.
    """
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            try:
                return _call_with_anthropic_sdk(question, max_new_tokens)
            except ModuleNotFoundError:
                # anthropic package not installed; use raw HTTP
                return _call_with_requests(question, max_new_tokens)

        except Exception as e:
            last_err = e
            sleep_s = min(2 ** (attempt - 1), 20)
            print(f"[WARN] API call failed (attempt {attempt}/{max_retries}): {e}")
            time.sleep(sleep_s)

    print(f"[ERROR] Exhausted retries. Last error: {last_err}")
    return ""  # keep blank so resume logic can retry later

if __name__ == "__main__":
    # User input for scalability (same structure)
    selected_language = input("Enter the language to process (or 'all' for all languages): ").strip().capitalize()
    num_samples = int(input("Enter the number of samples per language (-1 for all): "))

    if selected_language.lower() == "all":
        process_languages = languages
    else:
        process_languages = [selected_language]

    # Output directory
    output_dir = "claude_3_haiku_inference_testdataset"
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
        df["Test Response"] = ""  # Initialize column

        new_filename = f"{lang}_claude_3_haiku_inference_data.csv"
        save_path = os.path.join(output_dir, new_filename)

        # Resume if file exists (same logic)
        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path)
            if "Test Response" in existing_df.columns:
                completed_rows = existing_df[existing_df["Test Response"].notna()].shape[0]
                if completed_rows > 0:
                    print(f"Resuming from {completed_rows} completed rows in {new_filename}")
                    df.iloc[:completed_rows] = existing_df.iloc[:completed_rows]

        # Generate responses row by row with progressive saving
        for idx in range(len(df)):
            if df.at[idx, "Test Response"]:  # Skip completed
                continue

            question = df.at[idx, "Question"]
            print(f"Generating response for row {idx+1}/{process_rows} in {lang}...")
            test_response = generate_response(question, max_new_tokens=1024)
            df.at[idx, "Test Response"] = test_response

            # Save after each generation
            df.to_csv(save_path, index=False)

        print(f"Completed and saved: {save_path}")
