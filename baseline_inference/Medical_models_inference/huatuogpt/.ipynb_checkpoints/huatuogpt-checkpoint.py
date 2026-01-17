import os
os.environ['HF_HOME'] = '/scratch/reh6ed/hf_cache'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'fork'  # Try fork for less overhead; falls back to spawn if CUDA early
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Reduce frag
import pandas as pd
from vllm import LLM, SamplingParams
import argparse
import time
import re
from tqdm import tqdm
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # For chat template & 4-bit fallback

# ----------------------------- Argument Parsing -----------------------------
parser = argparse.ArgumentParser(
    description="Generate test responses for multilingual medical QA datasets using HuatuoGPT-o1-70B (vLLM + 2 GPUs)."
)
parser.add_argument(
    "--language",
    type=str,
    default="all",
    help="Specific language to process (e.g., 'French') or 'all' to process every language. Case-insensitive."
)
parser.add_argument(
    "--model_name",
    type=str,
    default="FreedomIntelligence/HuatuoGPT-o1-70B",
    help="Hugging Face model name (default: HuatuoGPT-o1-70B)."
)
parser.add_argument(
    "--tensor_parallel_size",
    type=int,
    default=2,
    help="Number of GPUs for vLLM tensor parallelism (default: 2)."
)
parser.add_argument(
    "--gpu_memory_utilization",
    type=float,
    default=0.18,  # Low default to fit worker-detected mem (~25 GiB)
    help="vLLM GPU memory utilization fraction (0.1-0.9; default: 0.18 for low-mem stability)."
)
parser.add_argument(
    "--max_model_len",
    type=int,
    default=4096,
    help="Max context length (default: 4096 to reduce KV cache overhead)."
)
parser.add_argument(
    "--quantization",
    type=str,
    default=None,
    choices=[None, "awq", "gptq"],
    help="Quantization method (if model supports; e.g., 'awq' for ~50% memory savings)."
)
parser.add_argument(
    "--use_transformers",
    action="store_true",
    help="Fallback to transformers + 4-bit bitsandbytes (slower, but works if vLLM OOMs)."
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=None,
    help="Maximum number of samples to process per language (for testing). Default: None (all)."
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Delete existing output CSVs and start fresh."
)
parser.add_argument(
    "--delay",
    type=float,
    default=0.0,
    help="Delay (seconds) between batches (rarely needed with vLLM)."
)
parser.add_argument(
    "--save_every",
    type=int,
    default=5,
    help="Save intermediate CSV every N questions (default: 5)."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,  # Low default for mem safety
    help="Batch size for parallel generation (default: 1; tune based on GPU mem)."
)
args = parser.parse_args()

if __name__ == '__main__':
    import torch  # Delayed import to avoid early CUDA init

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

    # ----------------------------- Pre-Load GPU Memory Check -----------------------------
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3) - torch.cuda.memory_reserved(i) / (1024**3)
            target_mem = args.gpu_memory_utilization * torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: Free ~{free_mem:.1f} GiB / Total {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GiB. Target: {target_mem:.1f} GiB.")
            if free_mem < target_mem * 0.9:  # 10% buffer
                print(f"WARNING: GPU {i} may OOM. Try --gpu_memory_utilization {free_mem / (torch.cuda.get_device_properties(i).total_memory / (1024**3)):.2f} or free memory.")

    # ----------------------------- Load Model -----------------------------
    print(f"Loading model {args.model_name} with {'transformers (4-bit)' if args.use_transformers else 'vLLM'} on {args.tensor_parallel_size} GPU(s)... (this may take a while)")
    llm = None
    tokenizer = None
    if args.use_transformers:
        # 4-bit fallback: transformers + bitsandbytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def generate_with_transformers(prompts, sampling_params):
            responses = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_model_len).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=sampling_params.max_tokens,
                        temperature=sampling_params.temperature,
                        top_p=sampling_params.top_p,
                        repetition_penalty=sampling_params.repetition_penalty,
                        do_sample=sampling_params.temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                responses.append(response)
            return [RequestOutput(outputs=[GeneratedText(text=r)]) for r in responses]  # Mock vLLM format

        class RequestOutput:
            def __init__(self, outputs):
                self.outputs = outputs

        class GeneratedText:
            def __init__(self, text):
                self.text = text

    else:
        try:
            llm = LLM(
                model=args.model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype="auto",
                trust_remote_code=True,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                quantization=args.quantization,
                enforce_eager=True,  # Disable graphs to reduce mem overhead
                disable_log_stats=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        except RuntimeError as e:
            if "Free memory on device" in str(e) or "Engine core initialization failed" in str(e):
                print(f"\nERROR: GPU memory insufficient for vLLM. Details: {e}")
                print("\nTRY: python huatuogpt.py --use_transformers --language Spanish --num_samples 10")
                raise
            raise

    # Deterministic greedy decoding
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048,
        repetition_penalty=1.1,
        stop=["<|eot_id|>"]
    )

    # ----------------------------- Response Generation Function -----------------------------
    def get_model_responses(queries: list[str], lang: str) -> list[str]:
        """
        Batch-generate responses using o1-style reasoning.
        Extracts ONLY the final answer after the LAST '## Final Response'.
        Retries 3x on failure/empty/no-marker (per batch).
        """
        system_prompt = f"""You are an expert medical doctor fluent in {lang}.
Analyze the patient case or question carefully in {lang}.
Think step by step in {lang} only: reflect on possible errors, explore alternatives, refine your reasoning.
Then provide the final answer in {lang} only.
Always structure your response exactly like this:
## Thinking
[your detailed reasoning here]
## Final Response
[your final answer only, no additional text]"""
        messages_template = [
            {"role": "system", "content": system_prompt},
        ]
        for attempt in range(3):
            try:
                prompts = []
                for query in queries:
                    messages = messages_template + [{"role": "user", "content": query}]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(prompt)
                if args.use_transformers:
                    outputs = generate_with_transformers(prompts, sampling_params)
                else:
                    outputs = llm.generate(prompts, sampling_params)
                responses = []
                for i, output in enumerate(outputs):
                    generated_text = output.outputs[0].text.strip()
                    query = queries[i]
                    # Remove accidental query echo
                    if generated_text.lower().startswith(query.lower()[:150]):
                        generated_text = generated_text[len(query):].strip()
                    # Extract after LAST '## Final Response'
                    matches = re.findall(r"## Final Response\s*([\s\S]*)", generated_text, re.DOTALL)
                    if matches:
                        response = matches[-1].strip()
                        if response and len(response) > 10:
                            responses.append(response)
                            continue
                    # Fallback to full text if valid
                    if generated_text and len(generated_text) > 10:
                        responses.append(generated_text)
                    else:
                        responses.append("")
                # Check for empties/short
                if all(len(r) > 10 for r in responses if r):
                    return responses
                print(f"Attempt {attempt+1}: {sum(1 for r in responses if len(r) <= 10)} empty/short in batch.")
                time.sleep(5)  # Longer sleep
            except Exception as e:
                print(f"Attempt {attempt+1} batch error: {e}")
                time.sleep(5)
        # Final fallback
        return ["FAILED_TO_GENERATE"] * len(queries)

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
        else:
            to_process_count = pending_count
        limit_str = f" (limited to {args.num_samples} for testing)" if (args.num_samples is not None and args.num_samples > 0) else ""
        print(f"Processing {to_process_count}/{original_pending} pending questions{limit_str} for {lang}...")
        # Batch processing for efficiency
        queries = []
        indices = []
        processed_this_run = 0
        pbar = tqdm(to_process, desc=f"{lang} progress")
        for idx in pbar:
            query = df.at[idx, 'Question']
            if not isinstance(query, str) or query.strip() == "":
                df.at[idx, 'Test_Response'] = "INVALID_QUERY"
                processed_this_run += 1
                continue
            queries.append(query)
            indices.append(idx)
            # Process batch when full
            if len(queries) == args.batch_size:
                responses = get_model_responses(queries, lang)
                for j, resp in enumerate(responses):
                    df.at[indices[j], 'Test_Response'] = resp
                processed_this_run += len(queries)
                queries = []
                indices = []
                pbar.update(len(responses))
                if processed_this_run % args.save_every == 0 or processed_this_run >= to_process_count:
                    df.to_csv(temp_path, index=False)
                    if os.path.exists(output_path):
                        os.replace(temp_path, output_path)
                    else:
                        os.rename(temp_path, output_path)
                    print(f" Saved progress @{processed_this_run}/{to_process_count} → ./{output_path}")
        # Handle final partial batch
        if queries:
            responses = get_model_responses(queries, lang)
            for j, resp in enumerate(responses):
                df.at[indices[j], 'Test_Response'] = resp
            processed_this_run += len(queries)
            pbar.update(len(responses))
        # Final save
        df.to_csv(temp_path, index=False)
        if os.path.exists(output_path):
            os.replace(temp_path, output_path)
        else:
            os.rename(temp_path, output_path)
        successful = len(df[df['Test_Response'].notna() & (df['Test_Response'] != "FAILED_TO_GENERATE") & (df['Test_Response'] != "INVALID_QUERY")])
        print(f"✅ Finished {lang}! {successful}/{len(df)} valid responses.")
        print(f" Saved to: ./{output_path}\n")
        if args.delay > 0:
            time.sleep(args.delay)

    print(f"All done! Results are in: ./{output_base}/")
    print("Each CSV has: Question | Response | Test_Response (clean final answers only)")
    print("Re-run anytime — resumes automatically unless --overwrite is used.")
    print("Examples:")
    print(" Test 10 samples (vLLM low-mem): python huatuogpt.py --language French --num_samples 10 --overwrite --gpu_memory_utilization 0.18 --batch_size 1")
    print(" Test 10 samples (4-bit fallback): python huatuogpt.py --use_transformers --language French --num_samples 10 --overwrite")
    print(" Full run: python huatuogpt.py --language all --batch_size 2")



# import os
# os.environ['HF_HOME'] = '/scratch/reh6ed/hf_cache'
# os.environ['TRANSFORMERS_CACHE'] = '/scratch/reh6ed/hf_cache'

# import pandas as pd
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams
# import torch
# import argparse
# import time
# import re
# from tqdm import tqdm
# import glob

# # ----------------------------- Argument Parsing -----------------------------
# parser = argparse.ArgumentParser(description="Generate test responses for multilingual medical QA datasets using HuatuoGPT-o1-8B (vLLM + 2 GPUs).")
# parser.add_argument(
#     "--language",
#     type=str,
#     default="all",
#     help="Specific language to process (e.g., 'French') or 'all' to process every language. Case-insensitive."
# )
# parser.add_argument(
#     "--model_name",
#     type=str,
#     #default="FreedomIntelligence/HuatuoGPT-o1-8B",
#     default="FreedomIntelligence/HuatuoGPT-o1-70B",
#     help="Hugging Face model name (default: HuatuoGPT-o1-8B)." 
# )
# parser.add_argument(
#     "--tensor_parallel_size",
#     type=int,
#     default=2,
#     help="Number of GPUs for vLLM tensor parallelism (you have 2 GPUs → default 2). Set to 1 for single GPU."
# )
# parser.add_argument(
#     "--num_samples",
#     type=int,
#     default=None,
#     help="Maximum number of samples to process per language (for testing). Default: None → process all remaining."
# )
# parser.add_argument(
#     "--overwrite",
#     action="store_true",
#     help="If set, delete existing output CSVs and start completely fresh (ignores previous responses)."
# )
# parser.add_argument(
#     "--delay",
#     type=float,
#     default=0.0,
#     help="Optional delay (seconds) between queries (rarely needed with vLLM). Default 0."
# )
# parser.add_argument(
#     "--save_every",
#     type=int,
#     default=5,
#     help="Save intermediate CSV every N questions. Default 5."
# )
# args = parser.parse_args()

# # ----------------------------- Paths & Languages -----------------------------
# base_path = "/standard/AikyamLab/eric/Multilingual medical reasoning/conversational finetuning/test_dataset"
# model_folder_name = args.model_name.split("/")[-1] + "_results"
# output_base = model_folder_name
# os.makedirs(output_base, exist_ok=True)
# print(f"Outputs will be saved in ./{output_base}/ (current directory)")

# csv_paths = glob.glob(os.path.join(base_path, "*.csv"))
# languages = []
# for p in csv_paths:
#     fname = os.path.basename(p)
#     if fname.endswith(".csv") and "_with_test" not in fname and not fname.startswith("temp_"):
#         lang = fname.replace(".csv", "")
#         languages.append(lang)

# if not languages:
#     raise ValueError(f"No CSV files found in {base_path}")

# if args.language.lower() != "all":
#     target = args.language.strip().capitalize()
#     languages = [l for l in languages if l.lower() == target.lower()]
#     if not languages:
#         raise ValueError(f"Language '{args.language}' not found. Available: {', '.join(sorted([l.lower() for l in languages]))}")

# print(f"Found {len(languages)} languages: {', '.join(languages)}")
# if args.language.lower() != "all":
#     print(f"Processing only: {languages[0]} (for testing)")

# # ----------------------------- Load Model & Tokenizer (vLLM) -----------------------------
# print(f"Loading model {args.model_name} with vLLM on {args.tensor_parallel_size} GPU(s)... (this may take a while)")
# tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
# llm = LLM(
#     model=args.model_name,
#     tensor_parallel_size=args.tensor_parallel_size,  # FULLY utilizes your 2 GPUs
#     dtype="auto",
#     trust_remote_code=True,
#     max_model_len=8192  # Safe for long reasoning chains
# )

# # Deterministic greedy decoding (matches your original do_sample=False)
# sampling_params = SamplingParams(
#     temperature=0.0,
#     top_p=1.0,
#     max_tokens=2048,
#     repetition_penalty=1.1,
#     stop=["<|eot_id|>"]  # Clean stop
# )

# # ----------------------------- Response Generation Function -----------------------------
# def get_model_response(query: str, lang: str) -> str:
#     """
#     Triggers full o1-style reasoning (Thinking → Final Response) using official chat template.
#     Extracts ONLY the final answer after the LAST '## Final Response' for clean output.
#     Retries 3× on failure/empty/no-marker.
#     """
#     system_prompt = f"""You are an expert medical doctor fluent in {lang}.
# Analyze the patient case or question carefully in {lang}.
# Think step by step in {lang} only: reflect on possible errors, explore alternatives, refine your reasoning.
# Then provide the final answer in {lang} only.
# Always structure your response exactly like this:

# ## Thinking
# [your detailed reasoning here]

# ## Final Response
# [your final answer only, no additional text]"""

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": query}
#     ]

#     for attempt in range(3):
#         try:
#             prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             outputs = llm.generate([prompt], sampling_params)
#             generated_text = outputs[0].outputs[0].text.strip()

#             # Remove accidental query echo
#             if generated_text.lower().startswith(query.lower()[:150]):
#                 generated_text = generated_text[len(query):].strip()

#             # Extract everything after the LAST '## Final Response'
#             matches = re.findall(r"## Final Response\s*([\s\S]*)", generated_text, re.DOTALL)
#             if matches:
#                 response = matches[-1].strip()
#                 if response and len(response) > 10:
#                     return response

#             # If no marker found, return the full generated text (no fallback tag)
#             if generated_text and len(generated_text) > 10:
#                 return generated_text

#             print(f"Attempt {attempt+1}: Empty/short response (len={len(generated_text)}).")
#             time.sleep(2)

#         except Exception as e:
#             print(f"Attempt {attempt+1} error: {e}")
#             time.sleep(2)

#     return "FAILED_TO_GENERATE"

# # ----------------------------- Main Processing Loop -----------------------------
# for lang in languages:
#     input_path = os.path.join(base_path, f"{lang}.csv")
#     output_path = os.path.join(output_base, f"{lang}_with_test_responses.csv")
#     temp_path = os.path.join(output_base, f"temp_{lang}_with_test_responses.csv")

#     if args.overwrite and os.path.exists(output_path):
#         print(f"--> Overwrite enabled: Removing existing {output_path}")
#         os.remove(output_path)

#     if os.path.exists(output_path):
#         df = pd.read_csv(output_path)
#         print(f"Resuming existing file for {lang} ({len(df)} rows loaded)")
#     else:
#         df = pd.read_csv(input_path)
#         print(f"Loaded fresh {lang}.csv ({len(df)} rows) from input path")

#     if 'Question' not in df.columns:
#         raise ValueError(f"{lang}.csv is missing 'Question' column")
#     if 'Test_Response' not in df.columns:
#         df['Test_Response'] = pd.NA

#     pending_mask = (
#         df['Test_Response'].isna() |
#         (df['Test_Response'] == "FAILED_TO_GENERATE") |
#         (df['Test_Response'] == "INVALID_QUERY")
#     )
#     to_process = df[pending_mask].index.tolist()
#     pending_count = len(to_process)

#     if pending_count == 0:
#         print(f"✅ {lang} already fully processed ({len(df)} responses). Skipping.\n")
#         continue

#     original_pending = pending_count
#     if args.num_samples is not None and args.num_samples > 0:
#         to_process = to_process[:args.num_samples]
    
#     to_process_count = len(to_process)
#     limit_str = f" (limited to {args.num_samples} for testing)" if (args.num_samples is not None and args.num_samples > 0) else ""
#     print(f"Processing {to_process_count}/{original_pending} pending questions{limit_str} for {lang}...")

#     processed_this_run = 0
#     for idx in tqdm(to_process, desc=f"{lang} progress"):
#         query = df.at[idx, 'Question']
#         if not isinstance(query, str) or query.strip() == "":
#             df.at[idx, 'Test_Response'] = "INVALID_QUERY"
#             processed_this_run += 1
#             continue

#         response = get_model_response(query, lang)
#         df.at[idx, 'Test_Response'] = response
#         processed_this_run += 1

#         if processed_this_run % args.save_every == 0 or processed_this_run == to_process_count:
#             df.to_csv(temp_path, index=False)
#             if os.path.exists(output_path):
#                 os.replace(temp_path, output_path)
#             else:
#                 os.rename(temp_path, output_path)
#             print(f"  Saved progress @{processed_this_run}/{to_process_count} → ./{output_path}")

#         if args.delay > 0:
#             time.sleep(args.delay)

#     df.to_csv(temp_path, index=False)
#     if os.path.exists(output_path):
#         os.replace(temp_path, output_path)
#     else:
#         os.rename(temp_path, output_path)

#     successful = len(df[df['Test_Response'].notna() & (df['Test_Response'] != "FAILED_TO_GENERATE") & (df['Test_Response'] != "INVALID_QUERY")])
#     print(f"✅ Finished {lang}! {successful}/{len(df)} valid responses.")
#     print(f"   Saved to: ./{output_path}\n")

# print(f"All done! Results are in: ./{output_base}/")
# print("Each CSV has: Question | Response | Test_Response (clean final answers only)")
# print("Re-run anytime — resumes automatically unless --overwrite is used.")
# print("Examples:")
# print("  Test 10 samples (fresh): python script.py --language French --num_samples 10 --overwrite --tensor_parallel_size 2")
# print("  Full run (all languages, MAX SPEED on 2 GPUs): python script.py --language all --tensor_parallel_size 2")