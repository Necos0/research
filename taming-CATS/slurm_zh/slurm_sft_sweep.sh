#!/bin/bash

#SBATCH --job-name=sft_sweep_agent
#SBATCH --account=iict-sp1.ebling.cl.uzh
#SBATCH --output=logs/%x_%j_out.log
#SBATCH --error=logs/%x_%j_err.log 
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80000
#SBATCH --time=04:00:00

# Environment setup
conda activate sft
module load gpu
nvidia-smi

wandb agent shtosti/keep-it-simple-src/n048lz73
