#!/bin/bash
#SBATCH --job-name=alpaca-lora-65b-en-pt-es-ca
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:2
#SBATCH --output=.slurm/alpaca-lora-65b-en-pt-es-ca.out
#SBATCH --error=.slurm/alpaca-lora-65b-en-pt-es-ca.err

# activate virtual environment
source ../venv/bin/activate

# transformers cache
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# report to wandb
export WANDB_PROJECT=alpaca-lora-mt

# resume an existing run in cases of failure
export WANDB_RESUME=auto
export WANDB_RUN_ID=8hplz9al

export WANDB__SERVICE_WAIT=300

srun torchrun --standalone --nproc_per_node=2 finetune.py configs/alpaca-lora-65b-en-pt-es-ca.yaml
