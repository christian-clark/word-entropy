#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --ntasks=24
#SBATCH --mem=100gb

# good enough for first-token entropy
##SBATCH --time=1:00:00
##SBATCH --ntasks=24
##SBATCH --mem=100gb

# settings for natstories, 64 samples, 8 samples/batch
##SBATCH --time=3:00:00
##SBATCH --ntasks=24
##SBATCH --mem=350gb


set -e

#SCRIPT=mc_word_renyi_entropy.py
#SCRIPT=first_token_renyi_entropy_restricted.py
#SCRIPT=first_token_renyi_entropy_unrestricted.py
SCRIPT=first_token_entropy.py
#INPUT=inputs/naturalstories.sentitems
INPUT=inputs/dundee.sentitems
#INPUT=inputs/naturalstories.first10.sentitems
#OUTPUT=outputs/ns.r05ftr.tokmeasures
#OUTPUT=outputs/ns.new.r05ftr.tokmeasures
OUTPUT=outputs/dd.new.ftr.tokmeasures
MODEL=gpt2


#SAMPLES=64
#SAMPLES=8
#SAMPLES_PER_BATCH=8
#SAMPLES_PER_BATCH=4
#ALPHA=0.5
ALPHA=1

source ~/miniconda3/etc/profile.d/conda.sh
#conda init
conda activate hf_env

set -x
date
#python3 $SCRIPT $INPUT $MODEL -s $SAMPLES -b $SAMPLES_PER_BATCH > $OUTPUT
#python3 $SCRIPT $INPUT $MODEL > $OUTPUT
python3 $SCRIPT $INPUT $MODEL -a $ALPHA > $OUTPUT
#python3 $SCRIPT $INPUT $MODEL --unrestricted -a $ALPHA > $OUTPUT
#python3 $SCRIPT $INPUT $MODEL -s $SAMPLES -b $SAMPLES_PER_BATCH -c 32 > $OUTPUT
date

#SCRIPT=first_token_entropy_restricted.py
#SCRIPT=first_token_entropy_unrestricted.py
#SCRIPT=first_token_renyi_entropy_restricted.py
#SCRIPT=first_token_renyi_entropy_unrestricted.py
#SCRIPT=mc_word_entropy.py
#SCRIPT=mc_word_renyi_entropy.py
#SCRIPT=sample_word_entropy.py
#SCRIPT=oh_scripts/mc_word_renyi_entropy.py

#INPUT=inputs/dundee.sentitems
#INPUT=inputs/dundee.${i}of8.sentitems
#INPUT=inputs/dundee.1of8.sentitems
#INPUT=inputs/dundee.2of8.sentitems
#INPUT=inputs/dundee.3of8.sentitems
#INPUT=inputs/dundee.4of8.sentitems
#INPUT=inputs/dundee.5of8.sentitems
#INPUT=inputs/dundee.6of8.sentitems
#INPUT=inputs/dundee.7of8.sentitems
#INPUT=inputs/dundee.8of8.sentitems
#INPUT=inputs/hello.sentitems
#INPUT=inputs/naturalstories.first100.sentitems
#INPUT=inputs/naturalstories.first44.sentitems
#INPUT=inputs/naturalstories.first10.sentitems
#INPUT=inputs/naturalstories.first1.sentitems
#INPUT=inputs/naturalstories.sentitems
#INPUT=inputs/naturalstories.1of2.sentitems
#INPUT=inputs/naturalstories.2of2.sentitems
#INPUT=inputs/test2.sentitems
#INPUT=inputs/test3.sentitems
#INPUT=inputs/test_multiple.sentitems
#INPUT=inputs/test.sentitems

#OUTPUT=outputs/naturalstories.s64.b8.entropy
#OUTPUT=outputs/naturalstories.first100.s16.b8.entropy
#OUTPUT=outputs/naturalstories.first44.s24.entropy
#OUTPUT=outputs/naturalstories.first10.entropy
#OUTPUT=outputs/naturalstories.first1.entropy
#OUTPUT=outputs/naturalstories.first1.s128.entropy
#OUTPUT=outputs/naturalstories.first1.s64.entropy
#OUTPUT=outputs/test.entropy
#OUTPUT=outputs/naturalstories.firsttoken.entropy
#OUTPUT=outputs/naturalstories.firsttokenrest.entropy
#OUTPUT=outputs/dundee.1of4.entropy
