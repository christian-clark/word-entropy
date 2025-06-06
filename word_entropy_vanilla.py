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

NUM_SAMPLES = 5
DEBUG = True

def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG:", *args, **kwargs, file=sys.stderr)

def get_space_subword_idx(tokenizer):
    space_idx = []
    subword_idx = []

    for token, idx in tokenizer.vocab.items():
        if token.startswith("Ġ"):
            space_idx.append(idx)
        else:
            subword_idx.append(idx)

    return space_idx, subword_idx

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


def print_entropies(entropies, ids, word_final, tokenizer):
    assert len(ids) == len(word_final)
    curr_toks = list()
    curr_word_ix = 0
    # skip initial id and word_final, which are for BOS token
    for id, final in zip(ids[1:], word_final[1:]):
        tok = tokenizer.convert_ids_to_tokens(id)
        curr_toks.append(tok)
        if final:
            entropy = entropies[curr_word_ix].item()
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
    #ctx_size = model.config.max_position_embeddings
    bos_id = model.config.bos_token_id
    space_ixs, subword_ixs = get_space_subword_idx(tokenizer)

    print("word entropy")

    for story in stories:
        #words.extend(story.split(" "))
        tokenizer_output = tokenizer(story)
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask

        # these tokenizers do not append bos_id by default
        if "gpt" in model_variant or "pythia" in model_variant:
            ids = [bos_id] + ids
            attn = [1] + attn

        word_final = list()
        for i in range(len(ids) - 1):
            curr_id = ids[i]
            next_id = ids[i+1]
            if curr_id == bos_id or next_id in space_ixs:
                word_final.append(True)
            else:
                word_final.append(False)

        # technically this should be there for the final word, but we don't
        # care about its output entropy
        word_final.append(True)
        printDebug(word_final)

        printDebug(ids)

        # TODO parallelize across samples
        story_log_probs = list()
        for sample_ix in range(NUM_SAMPLES):
            printDebug(f"\n==== SAMPLE {sample_ix+1} ====")

            sample_log_probs = list()

            for i in range(len(ids) - 1):
                if not word_final[i]: continue
                printDebug()
                curr_log_prob = 0
                ids_curr = ids[:i+1]
                ids_curr_t = torch.tensor(ids_curr).unsqueeze(0)
                attn_curr = attn[:i+1]
                attn_curr_t = torch.tensor(attn_curr).unsqueeze(0)
                model_output = model(input_ids=ids_curr_t, attention_maks=attn_curr_t)
                all_logits = model_output.logits.squeeze(0)
                logits = all_logits[-1]
                # TODO use KV cache
                # for the first sampling step, only sample over tokens that
                # are the beginning of a new word
                space_logits = logits[space_ixs]
                space_probs = softmax(space_logits)
                sampled_ix = torch.multinomial(space_probs, 1, replacement=True)[0]
                sample_prob = space_probs[sampled_ix]
                printDebug("sample prob:", sample_prob)
                curr_log_prob += torch.log2(space_probs[sampled_ix]).item()
                # indices should be relative to the whole vocabulary
                sampled_ix = space_ixs[sampled_ix]
                printDebug("sampled_ix:", sampled_ix)
                printDebug("context:", tokenizer.convert_ids_to_tokens(ids_curr))
                printDebug("sample:", tokenizer.convert_ids_to_tokens(sampled_ix))

                ids_curr = ids_curr + [sampled_ix]
                ids_curr_t = torch.tensor(ids_curr).unsqueeze(0)
                attn_curr = attn_curr + [1]
                attn_curr_t = torch.tensor(attn_curr).unsqueeze(0)
                model_output = model(input_ids=ids_curr_t, attention_maks=attn_curr_t)
                logits = model_output.logits.squeeze(0)[-1]
                probs = softmax(logits)
                # for subsequent sampling steps, options are subword tokens
                # EOW. EOW sums over all space tokens
                subword_probs = probs[subword_ixs]
                space_probs = probs[space_ixs]
                eow_prob = torch.sum(space_probs, dim=0, keepdim=True)
                printDebug("EOW prob:", eow_prob.item())
                probs = torch.cat([subword_probs, eow_prob], dim=0)
                sampled_ix = torch.multinomial(probs, 1, replacement=True)[0].item()
                sample_prob = probs[sampled_ix]
                printDebug("sample prob:", sample_prob)
                curr_log_prob += torch.log2(sample_prob).item()
                if sampled_ix == len(probs)-1:
                    eow = True
                    printDebug("EOW chosen")
                else:
                    eow = False
                while not eow:
                    sampled_ix = subword_ixs[sampled_ix]
                    printDebug("next sample:", tokenizer.convert_ids_to_tokens(sampled_ix))
                    ids_curr = ids_curr + [sampled_ix]
                    ids_curr_t = torch.tensor(ids_curr).unsqueeze(0)
                    attn_curr = attn_curr + [1]
                    attn_curr_t = torch.tensor(attn_curr).unsqueeze(0)
                    model_output = model(input_ids=ids_curr_t, attention_maks=attn_curr_t)
                    logits = model_output.logits.squeeze(0)[-1]
                    probs = softmax(logits)
                    # for subsequent sampling steps, options are subword tokens
                    # EOW. EOW sums over all space tokens
                    subword_probs = probs[subword_ixs]
                    space_probs = probs[space_ixs]
                    eow_prob = torch.sum(space_probs, dim=0, keepdim=True)
                    printDebug("EOW prob:", eow_prob.item())
                    probs = torch.cat([subword_probs, eow_prob], dim=0)
                    sampled_ix = torch.multinomial(probs, 1, replacement=True)[0].item()
                    sample_prob = probs[sampled_ix]
                    printDebug("sample prob:", sample_prob)
                    curr_log_prob += torch.log2(sample_prob).item()
                    if sampled_ix == len(probs)-1:
                        eow = True
                        printDebug("EOW chosen")
                    else:
                        eow = False
                sample_log_probs.append(curr_log_prob)

                printDebug("DONE SAMPLING")
            story_log_probs.append(sample_log_probs)   

        printDebug("story log probs:")
        printDebug(story_log_probs)

        # dim: samples x words
        story_log_probs = torch.tensor(story_log_probs)
        assert story_log_probs.shape[0] == NUM_SAMPLES
        story_entropies = -torch.mean(story_log_probs, dim=0)
        printDebug("story entropies:")
        printDebug(story_entropies)
        print_entropies(story_entropies, ids, word_final, tokenizer)


if __name__ == "__main__":
    main()
