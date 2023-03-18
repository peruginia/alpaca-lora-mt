#!/bin/bash
#SBATCH --job-name=alpaca-lora-7b
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:4
#SBATCH --output=.slurm/alpaca-lora-7b.out
#SBATCH --error=.slurm/alpaca-lora-7b.err

# activate virtual environment
source ../venv/bin/activate

# transformers cache
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# report to wandb
export WANDB_PROJECT=alpaca-lora-mt

# resume an existing run in cases of failure
export WANDB_RESUME=auto

srun torchrun --standalone --nproc_per_node=4 finetune.py
