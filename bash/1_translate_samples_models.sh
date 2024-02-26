#!/bin/bash

#SBATCH --job-name=gnt_samples
#SBAATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000MB
#SBATCH --time=24:00:00
#SBATCH --account=attanasiog
#SBATCH --output=./logs/slurm-%A.out
#SBATCH --partition=compute

export TOKENIZERS_PARALLELISM=true

if [ -n "$SLURM_JOB_ID" ]; then
    module load miniconda3
    source /home/AttanasioG/.bashrc
    conda activate py310
fi

#MODEL="facebook/nllb-200-3.3B"
#MODEL_ID="nllb-200-3.3B"
#MODEL="Helsinki-NLP/opus-mt-en-de"
#MODEL_ID="opus-mt"
#MODEL="google/flan-t5-xxl"
#MODEL_ID="flan-t5-xxl"
#MODEL="meta-llama/Llama-2-70b-chat-hf"
#MODEL_ID="Llama-2-70b-chat-hf"
# MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
# MODEL_ID="Mixtral-8x7B-Instruct-v0.1"
MODEL="gpt-3.5-turbo"
# MODEL="gpt-4"
MODEL_ID=${MODEL}
PROMPT_TEMPLATE="instruction"
FILE_TO_TRANSLATE="./results/terms_v3_sample_europarl_plurals.json"
OUTPUT_FILE="./results/translations/terms_v3_sample_europarl_plurals_${MODEL_ID}.json"

echo "STARTING TRANSLATION"
python src/translate.py \
    --model_name_or_path ${MODEL} \
    --samples_file ${FILE_TO_TRANSLATE} \
    --output_file ${OUTPUT_FILE} \
    --prompt_template ${PROMPT_TEMPLATE}
echo "ENDING..."

