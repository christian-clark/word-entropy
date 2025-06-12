#!/bin/bash
#SBATCH -A PAS1200
#SBATCH --time=4:00:00
#SBATCH --ntasks=24
#SBATCH --mem=100gb


set -e

SCRIPT=/users/PAS2157/ceclark/git/word-entropy/mc_word_entropy.py
NAME=test
INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/${NAME}.sentitems
OUTPUT=/users/PAS2157/ceclark/git/word-entropy/entropies/${NAME}.entropy
MODEL=gpt2

source ~/miniconda3/etc/profile.d/conda.sh
#conda init
conda activate transformers

set -x
date
python3 $SCRIPT $INPUT $MODEL > $OUTPUT
date
