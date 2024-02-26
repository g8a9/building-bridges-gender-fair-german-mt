#!/bin/bash

#SBATCH --job-name=match_wiki
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64000MB
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --output=./logs/slurm-%A.out
#SBATCH --account=attanasiog

CORPUS="wikipedia"
# CORPUS="europarl"

if [ -n "$SLURM_JOB_ID" ]; then
    module load miniconda3
    source /home/AttanasioG/.bashrc
    conda activate py310
fi

python src/sample_from_corpus.py \
    --seed_file "./data/terms_v3_plural.csv" \
    --output_file "./data/terms_v3_sample_wikipedia_plurals.json" \
    --corpus ${CORPUS} \
    --n_samples 5 \
    --num_workers 10 \
    --context_length 2 \
    --target_col "English" \
    --run_parallel_jobs "True"
