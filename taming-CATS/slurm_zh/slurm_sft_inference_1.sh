#!/bin/bash

#SBATCH --job-name=sft_inference_multi_1

#SBATCH --account=iict-sp1.ebling.cl.uzh
#SBATCH --output=logs/%x_%j_out.log
#SBATCH --error=logs/%x_%j_err.log 
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=60000
#SBATCH --time=04:30:00              # Max runtime (hh:mm:ss)

conda activate sft
module load gpu
nvidia-smi

# Run the inference script
bash ./run_sft_inference_1.sh