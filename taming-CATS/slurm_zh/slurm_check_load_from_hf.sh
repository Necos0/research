#!/bin/bash

#SBATCH --job-name=check_load_from_hf

#SBATCH --account=iict-sp1.ebling.cl.uzh
#SBATCH --output=logs/output/%x_%j.log   # Save standard output
#SBATCH --error=logs/error/%x_%j.log     # Save error logs
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --cpus-per-task=4            # Number of CPU cores
#SBATCH --mem=3G                     # Memory allocation (adjust as needed)
#SBATCH --time=0:10:00              # Max runtime (hh:mm:ss)
#SBATCH --export=ALL                 # Export all environment variables

# Load required modules (if applicable)
module load gpu

# Activate Conda environment
# conda init
conda activate sft

# Run the script
python ./src/check_load_from_hf.py