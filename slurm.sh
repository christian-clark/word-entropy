#!/bin/bash
#SBATCH -A PAS1200
#SBATCH --time=1:00:00
#SBATCH --ntasks=24
#SBATCH --mem=100gb


# settings for natstories, 64 samples, 8 samples/batch
##SBATCH --time=3:00:00
##SBATCH --ntasks=24
##SBATCH --mem=350gb


set -e

SCRIPT=/users/PAS2157/ceclark/git/word-entropy/first_token_entropy.py
INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.sentitems
OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/naturalstories.firsttoken.entropy
MODEL=gpt2

#SAMPLES=64
#SAMPLES_PER_BATCH=8

source ~/miniconda3/etc/profile.d/conda.sh
#conda init
conda activate transformers

set -x
date
python3 $SCRIPT $INPUT $MODEL > $OUTPUT
#python3 $SCRIPT $INPUT $MODEL -s $SAMPLES -b $SAMPLES_PER_BATCH > $OUTPUT
#python3 $SCRIPT $INPUT $MODEL -s $SAMPLES -b $SAMPLES_PER_BATCH -c 32 > $OUTPUT
date

#SCRIPT=/users/PAS2157/ceclark/git/word-entropy/mc_word_entropy.py
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/hello.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.first100.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.first44.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.first10.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.first1.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/test2.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/test3.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/test_multiple.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/test.sentitems
#OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/naturalstories.s64.b8.entropy
#OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/naturalstories.first100.s16.b8.entropy
#OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/naturalstories.first44.s24.entropy
#OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/naturalstories.first10.entropy
#OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/naturalstories.first1.entropy
#OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/naturalstories.first1.s128.entropy
#OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/naturalstories.first1.s64.entropy
#OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/test.entropy
#OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/naturalstories.firsttoken.entropy
