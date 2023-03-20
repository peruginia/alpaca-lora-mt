#!/bin/bash
#SBATCH --job-name=alpaca-lora-mt-demo
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:0
#SBATCH --output=.slurm/alpaca-lora-mt-demo.out
#SBATCH --error=.slurm/alpaca-lora-mt-demo.err

export CUDA_VISIBLE_DEVICES=0

# activate virtual environment
source ../venv/bin/activate

# transformers cache
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

srun torchrun --standalone --nproc_per_node=1 generate.py
