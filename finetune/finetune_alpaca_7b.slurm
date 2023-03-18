#!/bin/bash
#SBATCH --job-name=finetune_alpaca_7b
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:4
#SBATCH --output=.slurm/finetune_alpaca_7b.out
#SBATCH --error=.slurm/finetune_alpaca_7b.err

# activate virtual environment
source ../venv/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

srun torchrun --standalone --nproc_per_node=4 finetune.py
