<div align="center">
  <h1>CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning</h1>

  <!-- Link buttons (adds the "neat" look like your examples) -->
  <p>
    <a href="https://cure-med.github.io" target="_blank">
      <img alt="Website" src="https://img.shields.io/badge/Website-Project%20Page-blue">
    </a>
    <a href="https://arxiv.org/abs/2601.13262" target="_blank">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2601.13262-B31B1B">
    </a>
    <a href="https://huggingface.co/datasets/Aikyam-Lab/CUREMED-BENCH" target="_blank">
      <img alt="Hugging Face Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E">
    </a>
  </p>

  <p><em>by</em></p>

  <table>
    <tr>
      <td align="center" style="padding: 0 16px;">
        <strong>Eric Onyame</strong><sup>*</sup><br/>
        University of Virginia
      </td>
      <td align="center" style="padding: 0 16px;">
        <strong>Akash Ghosh</strong><sup>*</sup><br/>
        IIT-Patna
      </td>
      <td align="center" style="padding: 0 16px;">
        <strong>Subhadip Baidya</strong><br/>
        IIT-Patna
      </td>
      <td align="center" style="padding: 0 16px;">
        <strong>Sriparna Saha</strong><br/>
        IIT-Patna
      </td>
      <td align="center" style="padding: 0 16px;">
        <strong>Xiuying Chen</strong><br/>
        MBZUAI
      </td>
      <td align="center" style="padding: 0 16px;">
        <strong>Chirag Agarwal</strong><br/>
        University of Virginia
      </td>
    </tr>
  </table>

  <p><sup>*</sup>Equal contribution. <strong>Corresponding authors:</strong> Eric Onyame, Akash Ghosh</p>
</div>



<br/>

This repository hosts the codebase and dataset for <strong>CURE-Med</strong>, a framework for improving multilingual medical reasoning in large language models (LLMs). Below, we provide an overview of the project along with key training and implementation details.



## Overview
Large language models (LLMs) perform strongly on monolingual math and commonsense reasoning, but they remain unreliable for multilingual medical reasoning—limiting safe use in real-world, multilingual healthcare settings. To address this, we introduce <strong>CUREMED-BENCH</strong>, a high-quality multilingual medical reasoning benchmark of open-ended questions with a single verifiable answer, spanning 13 languages, including under-represented languages such as Amharic, Yoruba, and Swahili. Building on this benchmark, we propose <strong>CURE-MED</strong>, a curriculum-informed reinforcement learning framework that combines code-switching-aware supervised fine-tuning with Group Relative Policy Optimization to improve both logical correctness and language stability. Across 13 languages, CURE-MED consistently outperforms strong baselines and scales effectively, reaching 85.21% language consistency and 54.35% logical correctness at 7B parameters, and 94.96% language consistency and 70.04% logical correctness at 32B parameters. Overall, our results move toward more reliable and equitable multilingual medical reasoning with LLMs.


## Key Figure

<p align="center">
  <img src="figures/cure_med.png" alt="CURE-MED pipeline overview" width="900">
  <br/>
  <em><strong>Figure 1.</strong> CURE-MED pipeline: (A) clinically validated multilingual data curation (e.g., MedlinePlus), (B) code-switching-aware supervised fine-tuning of a Qwen2.5-Instruct backbone, and (C) GRPO-guided curriculum RL from high- to mid- to low-resource languages to improve logical correctness and language consistency.</em>
</p>

<p align="center">
  <sub>High-resolution PDF: <a href="figures/cure_med.pdf">Figure 1</a></sub>
</p>

<br/>

For full technical details and experiments, see the **[paper on arXiv](https://arxiv.org/abs/2601.13262)** and the **[project website](https://cure-med.github.io)**.


---

## Dataset

- **CUREMED-BENCH**: Provided in `data.zip`. The dataset contains open-ended medical reasoning questions with a single verifiable answer across **13 languages**.  
  Unzip `data.zip` before running training or evaluation.

- **Hugging Face**: CUREMED-BENCH is also available on Hugging Face:  
  https://huggingface.co/datasets/Aikyam-Lab/CUREMED-BENCH

---

## Repository Structure

- `baseline_inference/` — Baseline inference scripts for evaluation.
- `SFT/` — Code-switching-aware supervised fine-tuning (SFT) training pipeline.
- `SFT_Inference/` — Inference and evaluation for SFT checkpoints.
- `Curriculum_RFT/` — Curriculum-informed reinforcement learning / RFT training (GRPO-guided).
- `RFT_Inference/` — Inference and evaluation for RFT checkpoints.
- `figures/` — Figures used in the README and paper.
- `README.md` — Project documentation.
- `data.zip` — Packaged dataset release for local use.
- `datasets/` — Codeswitched dataset release for local use.

## Supervised Fine-Tuning (SFT)

SFT code is in `SFT/`:
- `code_switch_sft.py` — code-switching-aware SFT training script
- `deepspeed_zero3.yaml` — DeepSpeed ZeRO-3 config
- `code_switch_batch_script.sh` — Slurm batch script for launching SFT

### Environment
We ran SFT with **Python 3.11.13**. Ensure your environment includes:
`torch`, `transformers`, `datasets`, `trl`, `accelerate`, `deepspeed`.

### Data (SFT_data)
Provide SFT training files as JSONL under a directory like:
`/path/to/SFT_data/*.jsonl`

Each example must contain: `question`, `reasoning`, `answer`, `language`.

Set the dataset path in the batch script via:
`--data_dir="/path/to/SFT_data"`.

### Models & GPU Recommendations
We used Qwen2.5-Instruct variants: **1.5B, 3B, 7B, 14B, 32B**.
- **1.5B / ~4B**: recommended **4× A100**
- **7B / 14B / 32B**: recommended **≥ 8× A100**

### SFT Hyperparameters
- Optimizer: AdamW (β1=0.9, β2=0.999)
- LR: 1e-5 (cosine, warmup ratio 0.1), epochs: 3
- Effective batch size: 32, max seq length: 4096
- Precision: bf16, DeepSpeed ZeRO-3 + gradient checkpointing

### Run (Slurm)
Edit `SFT/code_switch_batch_script.sh`:
- set `base_model="Qwen/Qwen2.5-*-Instruct"`
- set `--data_dir="/path/to/SFT_data"`
- request GPUs via `#SBATCH --gres=gpu:a100:<N>`
- **match Accelerate processes to GPU count** (e.g., `--num_processes <N>`)

Submit:
sbatch SFT/code_switch_batch_script.sh

## Reinforcement Fine-Tuning (RFT / GRPO Curriculum)

RFT is implemented in `Curriculum_RFT/` as a **3-stage GRPO curriculum** over staged datasets:
- Datasets: `Curriculum_RFT/staged/{high,medium,low}/` (each contains `*.jsonl`)
- Training code: `Curriculum_RFT/Training_Stages/{Stage_one_training,Stage_two_training,Stage_three_training}/`

### Environment
We ran RFT with **Python 3.11.13** . All stages use **full fine-tuning**.

### Dataset format
Each JSONL example must include: `question`, `reasoning`, `answer`, `language`.

### Curriculum order (sequential checkpoints)
Run stages in this order (each stage initializes from the previous checkpoint):
1. **Stage 1 (High-resource)**: start from the **SFT checkpoint** + `staged/high/`
2. **Stage 2 (Medium-resource)**: start from **Stage 1 checkpoint** + `staged/medium/`
3. **Stage 3 (Low-resource)**: start from **Stage 2 checkpoint** + `staged/low/`

### Run RFT (Slurm)
Each stage provides a Slurm launcher script inside its stage folder. Update the script (or args) to point to the correct starting checkpoint and dataset path. 



<br/>

Note: Ensure accelerate --num_processes matches the number of GPUs requested in Slurm (e.g., 4 GPUs -> --num_processes 4). Output checkpoints are saved to the --output_dir specified in each stage script.
<br/>

**Inference:** The `SFT_Inference/` and `RFT_Inference/` folders contain scripts for running inference with the trained SFT and RFT checkpoints. Please update the relevant model/checkpoint and data paths in the scripts before running.


## Citation
If you find this repository useful, please consider citing our paper. (Citation coming soon.)





