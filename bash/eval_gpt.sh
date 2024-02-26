#!/bin/bash

#SBATCH --job-name=eval_string_matching
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=4000MB
#SBATCH --partition=compute
#SBATCH --output=./logs/slurm-%A.out
#SBATCH --account=attanasiog

module load miniconda3
source /home/AttanasioG/.bashrc
conda activate py310

MODELS=( \
    "deepl" \
    "flan-t5-xxl" \
    "google-translate" \
    "gpt-3.5-turbo" \
    "gpt-4" \
    "Llama-2-70b-chat-hf" \
    "nllb-200-3.3B" \
    "opus-mt" \
)

# for MODEL in "${MODELS[@]}"; do
echo "Evaluating model: ${MODEL}"
python ./src/evaluation/gpt.py \
    --translations_file "./data/human-evaluation_europarl_gpt-3.5.tsv" \
    --output_dir "./results/evaluation" \
    --model "gpt-3.5-turbo"
# done
