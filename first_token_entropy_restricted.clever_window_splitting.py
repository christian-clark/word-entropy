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

import sys, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXTokenizerFast, GPTNeoXForCausalLM


DEBUG = True

def debug(*args, **kwargs):
    if DEBUG:
        print("DEBUG:", *args, **kwargs, file=sys.stderr)


class Window:
    def __init__(self, input_ids, attn_mask, start_ix, is_first_window):
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.start_ix = start_ix
        self.is_first_window = is_first_window

    def __str__(self):
        s = "Window:\n\tInput_ids: {}\n\tAttn_mask: {}\n\tStart_ix: {}\n\tIs_first_window: {}".format(
            self.input_ids, self.attn_mask, self.start_ix,
            self.is_first_window
        )
        return s


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


def get_space_subword_idx(tokenizer):
    space_idx = []
    subword_idx = []

    # invert vocabulary dict so keys are ids and values are tokens
    # this allows space_idx and subword_idx to have consistent
    # orders across runs
    inverted_vocab = dict()
    for token, idx in tokenizer.vocab.items():
        inverted_vocab[idx] = token
        
    for idx in range(len(inverted_vocab)):
        token = inverted_vocab[idx]
        if token.startswith("Ġ"):
            space_idx.append(idx)
        else:
            subword_idx.append(idx)

    return torch.tensor(space_idx), torch.tensor(subword_idx)


def print_entropies(entropies, ids, word_final, tokenizer):
    debug("entropies:", entropies)
    debug("word_final:", word_final)
    assert len(ids) == len(word_final)
    curr_toks = list()
    curr_word_ix = 0
    # skip initial id and word_final, which are for BOS token
    for id, final in zip(ids[1:], word_final[1:]):
        tok = tokenizer.convert_ids_to_tokens(id)
        curr_toks.append(tok)
        if final:
            entropy = entropies[curr_word_ix]
            curr_word = tokenizer.convert_tokens_to_string(curr_toks).replace(" ", "")
            print(curr_word, entropy)
            curr_toks = list()
            curr_word_ix += 1


def main():
    stories = generate_stories(sys.argv[1])
    model_variant = sys.argv[2].split("/")[-1]

    if "gpt-neox" in model_variant:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(sys.argv[2])
    elif "gpt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2])
    elif "opt" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2])
    elif "pythia" in model_variant:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], revision=sys.argv[3], cache_dir=f"hf_models/{model_variant}_{sys.argv[3]}")

    else:
        raise ValueError("Unsupported LLM variant")

    if "pythia" in model_variant:
        model = GPTNeoXForCausalLM.from_pretrained(sys.argv[2], revision=sys.argv[3], cache_dir=f"hf_models/{model_variant}_{sys.argv[3]}")
    else:
        model = AutoModelForCausalLM.from_pretrained(sys.argv[2])

    # model.cuda()
    model.eval()
    softmax = torch.nn.Softmax(dim=-1)
    ctx_size = model.config.max_position_embeddings
    bos_id = model.config.bos_token_id
    space_ixs, subword_ixs = get_space_subword_idx(tokenizer)

    print("word entropy")
    for story in stories:
        #words.extend(story.split(" "))
        tokenizer_output = tokenizer(story)
        story_ids = tokenizer_output.input_ids
        story_attn = tokenizer_output.attention_mask

        # these tokenizers do not append bos_id by default
        if "gpt" in model_variant or "pythia" in model_variant:
            story_ids = [bos_id] + story_ids
            story_attn = [1] + story_attn

        story_word_final = list()
        story_word_initial = list()
        for i in range(len(story_ids) - 1):
            curr_id = story_ids[i]
            next_id = story_ids[i+1]
            if curr_id in space_ixs:
                story_word_initial.append(True)
            else:
                story_word_initial.append(False)
            if curr_id == bos_id or next_id in space_ixs:
                story_word_final.append(True)
            else:
                story_word_final.append(False)
        story_word_final.append(True)

        debug("story_ids[:20]:", story_ids[:20])
        debug("story_word_final[:20]:", story_word_final[:20])

        # split ids into subsequences, each of length <= 1/2 * ctx_size.
        # make sure that splits don't land mid-word
        # consecutive pairs of these subsequences will make up windows.
        # set up this way so context windows have 50% overlap
        half_window_start_ixs = [0]
        curr_start_ix = 0
        curr_end_ix = 0
        for ix, is_final in enumerate(story_word_final):
            if ix-curr_start_ix == ctx_size/2 - 1:
                curr_start_ix = curr_end_ix + 1
                half_window_start_ixs.append(curr_start_ix)
            elif is_final:
                curr_end_ix = ix
        half_window_start_ixs.append(len(story_word_final))
        
        # hacky fix for when the whole story fits in one context window
        if len(half_window_start_ixs) == 2:
            half_window_start_ixs.append(len(story_word_final))

        debug("half window start ixs [:20]:", half_window_start_ixs[:20])

        windows = list()
        is_first_window = True
        for ix in range(len(half_window_start_ixs)-2):
            w_start = half_window_start_ixs[ix]
            w_mid = half_window_start_ixs[ix+1]
            w_end = half_window_start_ixs[ix+2]
            # first window will use the entire window
            if ix == 0:
                start_ix = 0
            # later windows will use just the second half of the window
            else:
                start_ix = w_mid - w_start
            window = Window(
                input_ids=story_ids[w_start:w_end],
                attn_mask=story_attn[w_start:w_end],
                start_ix=start_ix,
                is_first_window=is_first_window
            )
            debug("adding window. start: {}, end: {}".format(w_start, w_end))
            windows.append(window)
            is_first_window = False

        story_entropies = list()
        for window in windows:
            ids = window.input_ids
            attn = window.attn_mask
            start_ix = window.start_ix
            is_first_window = window.is_first_window
            debug("curr window ids:", ids)

            debug("input ids type:", type(ids))
            model_output = model(
                input_ids=torch.tensor(ids).unsqueeze(0),
                attention_mask=torch.tensor(attn).unsqueeze(0)
            )

            logits = model_output.logits.squeeze(0)

            if is_first_window:
                # dim: V
                bos_logits = logits[0]

                debug("bos logits:", bos_logits)
                # first token after BOS is a subword_ix, not a space_ix
                bos_logits = bos_logits[subword_ixs]
                bos_probs = softmax(bos_logits)
                bos_log_probs = torch.log2(bos_probs)
                bos_entropy = torch.sum(-bos_probs*bos_log_probs, dim=0, keepdim=True)

                logits = logits[1:, space_ixs]
                probs = softmax(logits)
                log_probs = torch.log2(probs)
                entropies = torch.sum(-probs*log_probs, dim=1)
                window_entropies = torch.cat([bos_entropy, entropies], dim=0)
            else:
                # ignore tokens that aren't the beginning of a new word
                logits = logits[:, space_ixs]
                probs = softmax(logits)
                log_probs = torch.log2(probs)
                window_entropies = torch.sum(-probs*log_probs, dim=1)
            story_entropies.extend(window_entropies.tolist())
            word_entropies = list()
            for is_initial, entropy in zip(story_word_initial, story_entropies):
                if is_initial:
                    word_entropies.append(entropy)

        print_entropies(entropies, story_ids, story_word_final, tokenizer)


if __name__ == "__main__":
    main()
