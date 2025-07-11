#!/bin/bash
#SBATCH -A PAS1200
#SBATCH --time=3:00:00
#SBATCH --ntasks=24
#SBATCH --mem=350gb

# good enough for first-token entropy
##SBATCH --time=1:00:00
##SBATCH --ntasks=24
##SBATCH --mem=100gb

# settings for natstories, 64 samples, 8 samples/batch
##SBATCH --time=3:00:00
##SBATCH --ntasks=24
##SBATCH --mem=350gb


set -e

#SCRIPT=/users/PAS2157/ceclark/git/word-entropy/oh_scripts/mc_word_renyi_entropy.py
SCRIPT=/users/PAS2157/ceclark/git/word-entropy/mc_word_renyi_entropy.py
i=1
INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/dundee.${i}of8.sentitems
OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/renyi/dundee.temp${i}of8.oh.tokmeasures
MODEL=gpt2


#SAMPLES=64
SAMPLES=8
#SAMPLES_PER_BATCH=8
SAMPLES_PER_BATCH=4
#SEED=123

source ~/miniconda3/etc/profile.d/conda.sh
#conda init
conda activate transformers

set -x
date
python3 $SCRIPT $INPUT $MODEL -s $SAMPLES -b $SAMPLES_PER_BATCH > $OUTPUT
#python3 $SCRIPT $INPUT $MODEL > $OUTPUT
#python3 $SCRIPT $INPUT $MODEL -s $SAMPLES -b $SAMPLES_PER_BATCH -c 32 > $OUTPUT
date

#SCRIPT=/users/PAS2157/ceclark/git/word-entropy/first_token_entropy_restricted.py
#SCRIPT=/users/PAS2157/ceclark/git/word-entropy/first_token_entropy_unrestricted.py
#SCRIPT=/users/PAS2157/ceclark/git/word-entropy/first_token_renyi_entropy_restricted.py
#SCRIPT=/users/PAS2157/ceclark/git/word-entropy/first_token_renyi_entropy_unrestricted.py
#SCRIPT=/users/PAS2157/ceclark/git/word-entropy/mc_word_entropy.py
#SCRIPT=/users/PAS2157/ceclark/git/word-entropy/mc_word_renyi_entropy.py
#SCRIPT=/users/PAS2157/ceclark/git/word-entropy/sample_word_entropy.py
#SCRIPT=/users/PAS2157/ceclark/git/word-entropy/oh_scripts/mc_word_renyi_entropy.py

#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/dundee.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/dundee.${i}of8.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/dundee.1of8.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/dundee.2of8.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/dundee.3of8.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/dundee.4of8.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/dundee.5of8.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/dundee.6of8.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/dundee.7of8.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/dundee.8of8.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/hello.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.first100.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.first44.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.first10.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.first1.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.1of2.sentitems
#INPUT=/users/PAS2157/ceclark/git/word-entropy/inputs/naturalstories.2of2.sentitems
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
#OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/naturalstories.firsttokenrest.entropy
#OUTPUT=/users/PAS2157/ceclark/git/word-entropy/outputs/dundee.1of4.entropy
