#!/usr/bin/bash -l

#SBATCH --account=iict-sp1.ebling.cl.uzh
#SBATCH --gpus=1
#SBATCH --mem=2000
#SBATCH --time=00:05:00
#SBATCH --output=job_out.log
#SBATCH --error=job_error.log

conda activate sft
module load gpu
nvidia-smi

python ./src/WANDB_test.py
