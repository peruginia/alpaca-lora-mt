#!/bin/bash
#SBATCH --job-name=alpaca-lora-7b-demo
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:0
#SBATCH --output=.slurm/alpaca-lora-7b-demo.out
#SBATCH --error=.slurm/alpaca-lora-7b-demo.err

export CUDA_VISIBLE_DEVICES=0

# activate virtual environment
source ../venv/bin/activate

# transformers cache
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

srun python3 generate.py
