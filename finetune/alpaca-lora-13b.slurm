#!/bin/bash
#SBATCH --job-name=alpaca-lora-13b
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/alpaca-lora-13b.out
#SBATCH --error=.slurm/alpaca-lora-13b.err

# activate virtual environment
source ../venv/bin/activate

# transformers cache
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# report to wandb
export WANDB_PROJECT=alpaca-lora-mt

# resume an existing run in cases of failure
# export WANDB_RESUME=auto

for lang in en es eu; do
    srun torchrun --standalone --nproc_per_node=1 finetune.py configs/alpaca-lora-13b-"$lang".yaml
done
