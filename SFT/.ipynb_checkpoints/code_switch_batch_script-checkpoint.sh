#!/bin/bash -l
#SBATCH --job-name=bilingual_code_switch_qwen_sft_full
#SBATCH --output=logs/bilingual_code_switch_full_%j.out
#SBATCH --error=logs/bilingual_code_switch_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=
#SBATCH --mem=
#SBATCH --time=72:00:00
#SBATCH --account=
#SBATCH --mail-type=
#SBATCH --mail-user=

set -euo pipefail

# Clear any existing modules to avoid conflicts
module purge

# Load GCC for compatibility
module load gcc/11.4.0

# Reset Conda stack to avoid inherited broken state
unset CONDA_SHLVL
unset CONDA_PREFIX
unset CONDA_PROMPT_MODIFIER
for var in $(env | grep -E '^CONDA_PREFIX_[0-9]+=' | cut -d= -f1); do
    unset $var
done
unset CONDA_DEFAULT_ENV

# Activate Conda environment
source /home/reh6ed/miniconda3/etc/profile.d/conda.sh
export CONDA_NO_PLUGINS=true
conda activate grpo_env

# Set environment variables for caching and performance
export HF_HOME=/scratch/${USER}/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16  # Adjusted for better CPU utilization, especially with potential offloading
export NCCL_SOCKET_FAMILY=IPv4  # Suppress IPv6 socket warnings

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=============================================="
echo "Medical SFT 14B Multi-GPU Full Fine-tuning"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-4}"
echo "Start time: $(date)"
echo "=============================================="

# Print GPU info for debugging
nvidia-smi

# Quick Python check for CUDA
python - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
PY

# Training parameters
uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-1.5B-Instruct"
lr=1e-5
epochs=3
weight_decay=1e-4
micro_batch_size=4
gradient_accumulation_steps=4
max_steps=-1  # For full epochs
output_dir="bilingual_lang_switch_output_full_${uid}"

# Launch training with Accelerate and DeepSpeed
python -m accelerate.commands.launch \
    --config_file deepspeed_zero3.yaml \
    --num_processes 2 \
    --num_machines 1 \
    code_switch_sft.py \
    --block_size=4096 \
    --per_device_train_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --gradient_checkpointing=True \
    --num_train_epochs=${epochs} \
    --data_dir="/standard/AikyamLab/eric/Multilingual medical reasoning/conversational finetuning/NEW_ALGORITHM_3RD/Dataset/SFT_data" \
    --model_name=${base_model} \
    --warmup_ratio=0.1 \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --save_total_limit=1 \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.999 \
    --output_dir=${output_dir} \
    --push_to_hub=false \
    --max_steps=${max_steps} \
    --report_to "wandb"

echo "=============================================="
echo "Training completed"
echo "End time: $(date)"
echo "Output directory: ${output_dir}"
echo "=============================================="