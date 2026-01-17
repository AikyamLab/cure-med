#!/bin/bash -l
#SBATCH --job-name=medical_grpo_stage1_high
#SBATCH --output=logs/medical_grpo_stage1_high_%j.out
#SBATCH --error=logs/medical_grpo_stage1_high_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=
#SBATCH --mem=
#SBATCH --time=72:00:00

set -euo pipefail

# Activate YOUR conda environment directly
source /home/../miniconda3/etc/profile.d/conda.sh
conda activate grpo_env

# Set job-specific cache directory - this won't interfere with other jobs
export JOB_CACHE_DIR="/scratch/${USER}/hf_cache_${SLURM_JOB_ID}"
export HF_HOME="${JOB_CACHE_DIR}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

# Create job-specific cache directory
mkdir -p "${JOB_CACHE_DIR}"
echo "Using job-specific cache: ${JOB_CACHE_DIR}"

# Create logs directory
mkdir -p logs

# Set up cleanup trap to remove job-specific cache after completion
trap "echo 'Cleaning up job-specific cache...'; rm -rf ${JOB_CACHE_DIR}" EXIT

echo "=============================================="
echo "Medical GRPO Stage 1 High Resource Fine-tuning"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Cache Directory: ${JOB_CACHE_DIR}"
echo "Start: $(date)"
echo "=============================================="

nvidia-smi

# Training parameters
uid="$(date +%Y%m%d_%H%M%S)"
output_dir="stage_two_medium_output_${uid}"


# Get number of GPUs
GPUS_PER_NODE=2$(nvidia-smi -L | wc -l)
echo "Using ${GPUS_PER_NODE} GPUs"

# Run training with Accelerate and Deepspeed
python -m accelerate.commands.launch \
--config_file deepspeed_zero3.yaml \
--num_processes "$GPUS_PER_NODE" \
--num_machines 1 \
--main_process_ip 127.0.0.1 \
--machine_rank 0 \
--main_process_port 29502 \
--rdzv_backend c10d \
training_stage_two_medium.py \
--dataset_path "/standard/AikyamLab/eric/Multilingual medical reasoning/conversational finetuning/NEW_ALGORITHM_3RD/QWEN2.5_1.5B-INSTRUCT/Curriculum_RFT/staged/medium" \
--block_size 2048 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--num_train_epochs 1 \
--max_steps 500 \
--warmup_ratio 0.1 \
--bf16 True \
--eval_strategy "no" \
--logging_steps 1 \
--save_strategy "no" \
--lr_scheduler_type "cosine" \
--learning_rate 1e-6 \
--weight_decay 0.1 \
--adam_beta1 0.9 \
--adam_beta2 0.999 \
--beta 0.01 \
--output_dir "${output_dir}" \
--push_to_hub false \
--save_only_model true \
--report_to "wandb" \
--num_generations 16 \
--max_prompt_length 1024 \
--max_completion_length 1024 \
--optim "adamw_torch"



echo "=============================================="
echo "Training completed: $(date)"
echo "Output: ${output_dir}"
echo "Cache cleaned up: ${JOB_CACHE_DIR}"
echo "=============================================="