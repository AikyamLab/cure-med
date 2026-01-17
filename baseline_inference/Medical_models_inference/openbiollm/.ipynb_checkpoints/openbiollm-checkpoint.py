import os

os.environ['HF_HOME'] = '/scratch/reh6ed/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/reh6ed/hf_cache'

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import time
from tqdm import tqdm
import glob

# ----------------------------- Argument Parsing -----------------------------
parser = argparse.ArgumentParser(description="Generate test responses for multilingual medical QA datasets using OpenBioLLM-Llama3-8B (vLLM).")
parser.add_argument(
    "--language",
    type=str,
    default="all",
    help="Specific language to process (e.g., 'French') or 'all' to process every language. Case-insensitive."
)
parser.add_argument(
    "--model_name",
    type=str,
    default="aaditya/Llama3-OpenBioLLM-70B",
    help="Hugging Face model name (default: OpenBioLLM-Llama3-8B)."
)
parser.add_argument(
    "--tensor_parallel_size",
    type=int,
    default=2,
    help="Number of GPUs for vLLM tensor parallelism (you have 2 GPUs → default 2). Set to 1 for single GPU."
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=None,
    help="Maximum number of samples to process per language (for testing). Default: None → process all remaining."
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="If set, delete existing output CSVs and start completely fresh (ignores previous responses)."
)
parser.add_argument(
    "--delay",
    type=float,
    default=0.0,
    help="Optional delay (seconds) between queries (rarely needed with vLLM). Default 0."
)
parser.add_argument(
    "--save_every",
    type=int,
    default=5,
    help="Save intermediate CSV every N questions. Default 5."
)
args = parser.parse_args()

# ----------------------------- Paths & Languages -----------------------------
base_path = "/standard/AikyamLab/eric/Multilingual medical reasoning/conversational finetuning/test_dataset"
model_folder_name = args.model_name.split("/")[-1] + "_results"
output_base = model_folder_name
os.makedirs(output_base, exist_ok=True)
print(f"Outputs will be saved in ./{output_base}/ (current directory)")

csv_paths = glob.glob(os.path.join(base_path, "*.csv"))
languages = []
for p in csv_paths:
    fname = os.path.basename(p)
    if fname.endswith(".csv") and "_with_test" not in fname and not fname.startswith("temp_"):
        lang = fname.replace(".csv", "")
        languages.append(lang)

if not languages:
    raise ValueError(f"No CSV files found in {base_path}")

if args.language.lower() != "all":
    target = args.language.strip().capitalize()
    languages = [l for l in languages if l.lower() == target.lower()]
    if not languages:
        raise ValueError(f"Language '{args.language}' not found. Available: {', '.join(sorted([l.lower() for l in languages]))}")

print(f"Found {len(languages)} languages: {', '.join(languages)}")
if args.language.lower() != "all":
    print(f"Processing only: {languages[0]} (for testing)")

# ----------------------------- Load Model & Tokenizer (vLLM) -----------------------------
print(f"Loading model {args.model_name} with vLLM on {args.tensor_parallel_size} GPU(s)... (this may take a while)")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
llm = LLM(
    model=args.model_name,
    tensor_parallel_size=args.tensor_parallel_size,
    dtype="auto",
    trust_remote_code=True,
    max_model_len=8192
)

# Non-deterministic sampling: temperature=0.6, top_p=0.9 → diverse, high-quality outputs
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    max_tokens=2048,
    repetition_penalty=1.2,
    stop=["<|eot_id|>"]
)

# ----------------------------- Response Generation Function -----------------------------
def get_model_response(query: str, lang: str) -> str:
    """
    Manually builds the official Llama-3 chat prompt (fixes missing chat_template issue).
    Generates non-deterministic response with vLLM.
    """
    system_prompt = f"""You are an expert medical doctor fluent in {lang}.
You are given a medical question or patient case in {lang}.
Respond ENTIRELY in {lang} only. NEVER use English or any other language.
Provide a complete, accurate, professional medical answer directly.
Do not add any markers, tags, or extra text."""

    # Llama-3 special tokens 
    bos = tokenizer.bos_token if tokenizer.bos_token else "<|begin_of_text|>"
    eot = "<|eot_id|>"
    start_header = "<|start_header_id|>"
    end_header = "<|end_header_id|>"

    # Manual Llama-3 Instruct prompt construction
    prompt = bos
    for role, content in [("system", system_prompt), ("user", query)]:
        prompt += f"{start_header}{role}{end_header}\n\n{content.strip()}{eot}"
    prompt += f"{start_header}assistant{end_header}\n\n"

    for attempt in range(3):
        try:
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()

            if generated_text.lower().startswith(query.lower()[:150]):
                generated_text = generated_text[len(query):].strip()

            if generated_text and len(generated_text) > 10:
                return generated_text

            print(f"Attempt {attempt+1}: Empty/short response (len={len(generated_text)}).")
            time.sleep(2)

        except Exception as e:
            print(f"Attempt {attempt+1} error: {e}")
            time.sleep(2)

    return "FAILED_TO_GENERATE"

# ----------------------------- Main Processing Loop -----------------------------
for lang in languages:
    input_path = os.path.join(base_path, f"{lang}.csv")
    output_path = os.path.join(output_base, f"{lang}_with_test_responses.csv")
    temp_path = os.path.join(output_base, f"temp_{lang}_with_test_responses.csv")

    if args.overwrite and os.path.exists(output_path):
        print(f"--> Overwrite enabled: Removing existing {output_path}")
        os.remove(output_path)

    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        print(f"Resuming existing file for {lang} ({len(df)} rows loaded)")
    else:
        df = pd.read_csv(input_path)
        print(f"Loaded fresh {lang}.csv ({len(df)} rows) from input path")

    if 'Question' not in df.columns:
        raise ValueError(f"{lang}.csv is missing 'Question' column")
    if 'Test_Response' not in df.columns:
        df['Test_Response'] = pd.NA

    pending_mask = (
        df['Test_Response'].isna() |
        (df['Test_Response'] == "FAILED_TO_GENERATE") |
        (df['Test_Response'] == "INVALID_QUERY")
    )
    to_process = df[pending_mask].index.tolist()
    pending_count = len(to_process)

    if pending_count == 0:
        print(f"✅ {lang} already fully processed ({len(df)} responses). Skipping.\n")
        continue

    original_pending = pending_count
    if args.num_samples is not None and args.num_samples > 0:
        to_process = to_process[:args.num_samples]
    
    to_process_count = len(to_process)
    limit_str = f" (limited to {args.num_samples} for testing)" if (args.num_samples is not None and args.num_samples > 0) else ""
    print(f"Processing {to_process_count}/{original_pending} pending questions{limit_str} for {lang}...")

    processed_this_run = 0
    for idx in tqdm(to_process, desc=f"{lang} progress"):
        query = df.at[idx, 'Question']
        if not isinstance(query, str) or query.strip() == "":
            df.at[idx, 'Test_Response'] = "INVALID_QUERY"
            processed_this_run += 1
            continue

        response = get_model_response(query, lang)
        df.at[idx, 'Test_Response'] = response
        processed_this_run += 1

        if processed_this_run % args.save_every == 0 or processed_this_run == to_process_count:
            df.to_csv(temp_path, index=False)
            if os.path.exists(output_path):
                os.replace(temp_path, output_path)
            else:
                os.rename(temp_path, output_path)
            print(f"  Saved progress @{processed_this_run}/{to_process_count} → ./{output_path}")

        if args.delay > 0:
            time.sleep(args.delay)

    df.to_csv(temp_path, index=False)
    if os.path.exists(output_path):
        os.replace(temp_path, output_path)
    else:
        os.rename(temp_path, output_path)

    successful = len(df[df['Test_Response'].notna() & (df['Test_Response'] != "FAILED_TO_GENERATE") & (df['Test_Response'] != "INVALID_QUERY")])
    print(f"✅ Finished {lang}! {successful}/{len(df)} valid responses.")
    print(f"   Saved to: ./{output_path}\n")

print(f"All done! Results are in: ./{output_base}/")
print("Each CSV has: Question | Response | Test_Response")
print("Re-run anytime — resumes automatically unless --overwrite is used.")
print("Examples:")
print("  Test 10 samples (fresh): python script.py --language French --num_samples 10 --overwrite")
print("  Full run (all languages, 2 GPUs): python script.py --language all --tensor_parallel_size 2")