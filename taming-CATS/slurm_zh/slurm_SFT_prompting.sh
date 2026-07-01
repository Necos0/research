#!/bin/bash

#SBATCH --job-name=prompt_with_local_model

#SBATCH --account=iict-sp1.ebling.cl.uzh
#SBATCH --output=logs/%x_%j_output.log   # Save standard output
#SBATCH --error=logs/%x_%j_error.log     # Save error logs
#SBATCH --gpus=1
#SBATCH --mem=16000
#SBATCH --time=01:00:00              # Max runtime (hh:mm:ss)

conda activate sft
module load gpu
nvidia-smi

# Run the script
bash ./run_SFT_prompting.sh