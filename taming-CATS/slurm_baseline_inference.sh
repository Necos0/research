#!/bin/bash

#SBATCH --job-name="baseline_inference"
#SBATCH --output=logs/baseline_inference_%j.out
#SBATCH --error=logs/baseline_inference_%j.err
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-gpu=80GB
#SBATCH --time=04:00:00
#SBATCH --ntasks=1

which python3

# Activate the venv
source venv/bin/activate

nvidia-smi

# Run the baseline inference script
bash ./run_baseline_inference.sh