#!/bin/bash -l
#SBATCH --job-name=huatuo_inference_70b
#SBATCH --output=logs/huatuo_%j.out
#SBATCH --error=logs/huatuo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=80

set -euo pipefail

module purge # Clear any existing modules to avoid conflicts
module load gcc/11.4.0 # Load GCC 11.4.0 for GLIBCXX_3.4.29 support

source /home/reh6ed/miniconda3/etc/profile.d/conda.sh
export CONDA_NO_PLUGINS=true
conda activate grpo_env

# Caches & optimizations
export HF_HOME=/scratch/${USER}/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=10

mkdir -p logs

echo "=============================================="
echo "HuatuoGPT-o1-70B Inference on 4 GPUs (vLLM)"
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node : $(hostname)"
echo "GPU : ${SLURM_GPUS_ON_NODE:-4}"
echo "Start time : $(date)"
echo "=============================================="

nvidia-smi

python - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
PY

# Inference run - adjust args as needed (e.g., --language "all" for full run, --num_samples 10 for testing)
python openbiollm.py \
    --model_name "aaditya/Llama3-OpenBioLLM-70B" \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.75 \
    --max_model_len 4096 \
    --batch_size 2 \
    --language "all" \
    --save_every 5

echo "=============================================="
echo "Inference completed"
echo "End time: $(date)"
echo "Outputs saved in: ./HuatuoGPT-o1-70B_results/ (per-language CSVs)"
echo "=============================================="