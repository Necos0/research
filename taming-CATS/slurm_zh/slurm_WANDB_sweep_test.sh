#!/usr/bin/bash -l

#SBATCH --job-name=wandb_sweep_test
#SBATCH --account=iict-sp1.ebling.cl.uzh
#SBATCH --gpus=1
#SBATCH --mem=2000
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j_out.log
#SBATCH --error=logs/%x_%j_err.log

conda activate sft
module load gpu
nvidia-smi

python ./src/WANDB_sweep_test.py