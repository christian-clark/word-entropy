"""
Calculates LLM entropy from the following LLM families:

GPT-2 family:
"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

GPT-Neo family:
"EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
"EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"

OPT family:
"facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b",
"facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"

Pythia family:
"EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
"EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
each with checkpoints specified by training steps:
"step1", "step2", "step4", ..., "step142000", "step143000"
"""

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


DEBUG = False

def debug(*args, **kwargs):
    if DEBUG:
        print("DEBUG:", *args, **kwargs, file=sys.stderr)


class Window:
    def __init__(self, input_ids, output_ids, attn_mask, word_initial, start_ix):
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.attn_mask = attn_mask
        self.word_initial = word_initial
        self.start_ix = start_ix


def generate_stories(fn):
    stories = []
    f = open(fn)
    first_line = f.readline()
    assert first_line.strip() == "!ARTICLE"
    curr_story = ""

    for line in f:
        sentence = line.strip()
        if sentence == "!ARTICLE":
            stories.append(curr_story[:-1])
            curr_story = ""
        else:
            curr_story += line.strip() + " "

    stories.append(curr_story[:-1])
    return stories


def get_space_subword_idx(tokenizer, vocab_size):
    space_idx = []
    subword_idx = []

    # invert vocabulary dict so keys are ids and values are tokens
    # this allows space_idx and subword_idx to have consistent
    # orders across runs
    inverted_vocab = dict()
    for token, idx in tokenizer.vocab.items():
        inverted_vocab[idx] = token
        
    for idx in range(len(inverted_vocab)):
        # some non-English models have junk vocab idx that is never used
        # which will cause an indexing error if not filtered out
        if idx <= vocab_size-1:
            token = inverted_vocab[idx]
            if token.startswith("Ġ"):
                space_idx.append(idx)
            else:
                subword_idx.append(idx)

    return torch.tensor(space_idx), torch.tensor(subword_idx)


def main(args):
    print(args, file=sys.stderr)
    stories = generate_stories(args.sentitems)
    model_variant = args.model.split("/")[-1]
    os.environ["HF_HOME"] = "/scratch/bo2257/hf_cache"
    unrestricted = args.unrestricted
    alpha = args.alpha

    tokenizer = AutoTokenizer.from_pretrained(sys.argv[2])
    if "bne" in model_variant:
        vocab_size = 50261
    else:
        vocab_size = tokenizer.vocab_size
    space_ixs, subword_ixs = get_space_subword_idx(tokenizer, vocab_size)
    config = AutoConfig.from_pretrained(sys.argv[2])
    if "pythia" in model_variant:
        model = AutoModelForCausalLM.from_pretrained(sys.argv[2], revision=sys.argv[3])
    else:
        model = AutoModelForCausalLM.from_pretrained(sys.argv[2])
    model.cuda()
    model.eval()
    softmax = torch.nn.Softmax(dim=-1)

    if "gpt3-finnish" not in model_variant:
        ctx_size = config.max_position_embeddings
    else:
        ctx_size = 2048

    bos_id = config.bos_token_id
    no_bos_list = ["hebrew-gpt", "polyglot-ko", "rugpt3", "bne", "turkish-gpt2"]

    if any(item in model_variant for item in no_bos_list):
        prepend_bos = False
    else:
        prepend_bos = True
    print(f"prepend_bos: {prepend_bos}", file=sys.stderr)

    print("word entropy")
    for story in stories:
        tokenizer_output = tokenizer(story)
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask

        if prepend_bos:
            ids = [bos_id] + ids
            attn = [1] + attn

        if not prepend_bos and "bne" not in model_variant:
            first_tok = tokenizer.convert_ids_to_tokens([ids[0]])
            first_tok = tokenizer.convert_tokens_to_string(first_tok).replace(" ", "")
            print(f"adding: {first_tok}, 100.", file=sys.stderr)
            print(first_tok, 100.)

        # call initial BOS word-initial, as well as the token after
        # token right after BOS wil be in subword_ixs but is still word
        # intiial
        word_initial = [True, True]
        for idx in ids[2:]:
            if idx in space_ixs:
                word_initial.append(True)
            else:
                word_initial.append(False)

        debug("word_initial length:", len(word_initial))

        # split ids into subsequences, each of length <= 1/2 * ctx_size.
        # make sure that splits don't land mid-word
        # consecutive pairs of these subsequences will make up windows.
        # set up this way so context windows have 50% overlap
        half_window_start_ixs = list()
        half_window_start = 0
        last_word_start = 0
        for ix, is_initial in enumerate(word_initial):
            if is_initial:
                last_word_start = ix
            if ix-half_window_start == ctx_size/2:
                half_window_start_ixs.append(half_window_start)
                half_window_start = last_word_start
        half_window_start_ixs.append(half_window_start)
        half_window_start_ixs.append(len(word_initial))

        # edge cases for short input files that don't span multiple
        # windows
        if len(half_window_start_ixs) == 2:
            half_window_start_ixs.append(len(word_initial))

        windows = list()
        debug("half_window_start_ixs:", half_window_start_ixs)
        for ix in range(len(half_window_start_ixs)-2):
            w_start = half_window_start_ixs[ix]
            w_mid = half_window_start_ixs[ix+1]
            w_end = half_window_start_ixs[ix+2]
            # first window will use the entire window
            if ix == 0:
                start_ix = 0
            # later windows will use just the second half of the window
            else:
                start_ix = w_mid - w_start - 1
            window = Window(
                input_ids=ids[w_start:w_end-1],
                output_ids=ids[w_start+1:w_end],
                attn_mask=attn[w_start+1:w_end],
                word_initial=word_initial[w_start+1:w_end],
                start_ix=start_ix
            )
            debug("adding window. start: {}, end: {}".format(w_start, w_end))
            windows.append(window)

        first_window = True
        for window in windows:
            debug("input ids:", window.input_ids)
            input_ids = torch.tensor(window.input_ids).unsqueeze(0)
            output_ids = window.output_ids
            attn_mask = torch.tensor(window.attn_mask).unsqueeze(0)
            word_initial = window.word_initial
            start_ix = window.start_ix
            model_output = model(
                input_ids=input_ids.cuda(),
                attention_mask=attn_mask.cuda()
            )
            # discard initial BOS
            toks = tokenizer.convert_ids_to_tokens(output_ids)
            # dim: sents x V
            logits = model_output.logits.squeeze(0)
            if unrestricted:
                # shannon entropy
                probs = softmax(logits.double())
                if alpha == 1:
                    log_probs = torch.log2(probs)
                    entropies = torch.sum(-probs*log_probs, dim=1)
                else:
                    entropies = torch.log2(torch.sum(probs**alpha, dim=1)) / (1-alpha)
            else:
                if first_window:
                    # dim: V
                    bos_logits = logits[0]
                    # first token after BOS is a subword_ix, not a space_ix
                    bos_logits = bos_logits[subword_ixs]
                    bos_probs = softmax(bos_logits.double())
                    # shannon entropy
                    if alpha == 1:
                        bos_log_probs = torch.log2(bos_probs)
                        bos_entropy = torch.sum(-bos_probs*bos_log_probs, dim=0, keepdim=True)
                    else:
                        x = torch.sum(bos_probs**alpha, dim=0, keepdim=True)
                        bos_entropy = torch.log2(x) / (1-alpha)
                    logits = logits[1:, space_ixs]
                    probs = softmax(logits.double())
                    if alpha == 1:
                        log_probs = torch.log2(probs)
                        entropies = torch.sum(-probs*log_probs, dim=1)
                    else:
                        entropies = torch.log2(torch.sum(probs**alpha, dim=1)) / (1-alpha)
                    entropies = torch.cat([bos_entropy, entropies], dim=0)
                    first_window = False
                else:
                    # ignore tokens that aren't the beginning of a new word
                    logits = logits[:, space_ixs]
                    probs = softmax(logits.double())
                    if alpha == 1:
                        log_probs = torch.log2(probs)
                        entropies = torch.sum(-probs*log_probs, dim=1)
                    else:
                        entropies = torch.log2(torch.sum(probs**alpha, dim=1)) / (1-alpha)

            debug("start ix:", start_ix)
            debug("len(toks):", len(toks))
            for i in range(start_ix, len(toks)):
                cleaned_tok = tokenizer.convert_tokens_to_string([toks[i]]).replace(" ", "")
                is_initial = word_initial[i]
                if is_initial:
                    entropy = entropies[i].item()
                    print(cleaned_tok, entropy)
                else:
                    print(cleaned_tok, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sentitems", help="input file delimited with !ARTICLE")
    parser.add_argument("model", help="language model")
    parser.add_argument("-u", "--unrestricted", action="store_true", 
        help="take entropy over full vocab instead of space-initial tokens")
    parser.add_argument("-a", "--alpha", type=float, default=1.,
        help="alpha parameter for renyi entropy; alpha=1 gives shannon entropy")
    args = parser.parse_args()
    main(args)
