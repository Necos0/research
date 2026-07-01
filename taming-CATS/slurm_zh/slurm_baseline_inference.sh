#!/bin/bash

#SBATCH --job-name=baseline_inference

#SBATCH --account=iict-sp1.ebling.cl.uzh
#SBATCH --output=logs/%x_%j_out.log
#SBATCH --error=logs/%x_%j_err.log 
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80000
#SBATCH --time=10:00:00              # Max runtime (hh:mm:ss)

conda activate sft
module load gpu
nvidia-smi

# Run the baseline inference script
bash ./run_baseline_inference.sh
