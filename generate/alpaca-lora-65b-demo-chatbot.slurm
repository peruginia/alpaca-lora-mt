#!/bin/bash
#SBATCH --job-name=alpaca-lora-65b-demo-chatbot
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/alpaca-lora-65b-demo-chatbot.out
#SBATCH --error=.slurm/alpaca-lora-65b-demo-chatbot.err

# activate virtual environment
source ../venv/bin/activate

# transformers cache
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

BASE_URL=decapoda-research/llama-65b-hf
FINETUNED_CKPT_URL=HiTZ/alpaca-lora-65b-en-pt-es-ca-eu-gl-at
PORT=6065
BATCH_SIZE=1
API_OPEN=no
SHARE=no
GEN_CONFIG_PATH=Alpaca-LoRA-Serve/generation_config_default.yaml

srun python3 Alpaca-LoRA-Serve/app.py \
    --base_url $BASE_URL \
    --ft_ckpt_url $FINETUNED_CKPT_URL \
    --port $PORT \
    --batch_size $BATCH_SIZE \
    --api_open $API_OPEN \
    --share $SHARE \
    --gen_config_path $GEN_CONFIG_PATH