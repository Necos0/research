#!/bin/bash

#SBATCH --job-name=sft_finetune_peft

#SBATCH --account=iict-sp1.ebling.cl.uzh
#SBATCH --output=logs/%x_%j_out.log
#SBATCH --error=logs/%x_%j_err.log 
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80000
#SBATCH --time=3:00:00              # Max runtime (hh:mm:ss)

conda activate sft
module load gpu
nvidia-smi

# Run the script
bash ./run_sft_finetune_peft.sh