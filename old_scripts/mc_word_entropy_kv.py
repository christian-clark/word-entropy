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

import sys, torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoXTokenizerFast,
    GPTNeoXForCausalLM,
    DynamicCache
)

NUM_SAMPLES = 32
DEBUG = True

def printDebug(*args, **kwargs):
    if DEBUG:
        print("DEBUG:", *args, **kwargs, file=sys.stderr)

def get_space_subword_idx(tokenizer):
    space_idx = []
    subword_idx = []

    # invert vocabulary so keys are ids and values are tokens
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

    printDebug("space_idx:", space_idx[:20])
    printDebug("subword_idx:", subword_idx[:20])

    return torch.tensor(space_idx), torch.tensor(subword_idx)

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

            for i in range(0, len(ids) - 1):
                if i == 0:
                    id_start = torch.tensor(ids[:1]).unsqueeze(0)
                    attn_start = torch.tensor(attn[:1]).unsqueeze(0)
                    output = model(input_ids=id_start, attention_mask=attn_start, use_cache=True)
                else:
                    next_id = torch.tensor(ids[i:i+1])
                    printDebug("\nprocessing next token:", tokenizer.convert_ids_to_tokens(next_id))
                    output = model(
                        input_ids=next_id,
                        past_key_values=DynamicCache.from_legacy_cache(master_kv), use_cache=True
                    )
                master_kv = output.past_key_values
                
                if not word_final[i]: continue

                curr_log_prob = 0
                logits = output.logits.squeeze(dim=0).squeeze(dim=0)

                # GPT2 is not trained to put the initial whitespace
                # on the token immediately after BOS
                if ids[i] == bos_id:
                    first_logits = logits[subword_ixs]
                    # dim: samples x v
                    first_probs = softmax(first_logits)

                # all words coming after a word other than BOS will start
                # with whitespace
                else:
                    #printDebug("logits shape:", logits.shape)
                    #temp_sm = softmax(logits)
                    #printDebug("top10:", temp_sm.topk(10))
                    first_logits = logits[space_ixs]
                    # dim: samples x v
                    first_probs = softmax(first_logits)
                    # dim: batch x 1

                # TODO remove
                topk = first_probs.topk(10)[1]
                printDebug("topk next tokens (before ix conversion):", topk)
                if ids[i] == bos_id:
                    topk = subword_ixs[topk]
                else:
                    topk = space_ixs[topk]
                printDebug("topk next tokens (after ix conversion):", topk)
                # / TODO remove

                #sampled_ix = torch.multinomial(space_probs, 1, replacement=True)[0]
                ix = torch.multinomial(first_probs, 1)[0]
                sample_prob = first_probs[ix]
                curr_log_prob += torch.log2(sample_prob).item()
                # indices should be relative to the whole vocabulary
                if ids[i] == bos_id:
                    ix = torch.tensor([subword_ixs[ix]]) 
                else:
                    ix = torch.tensor([space_ixs[ix]]) 
                printDebug("sample:", ix, "logprob:", torch.log2(sample_prob))

                output = model(
                    input_ids=ix.unsqueeze(0),
                    past_key_values=DynamicCache.from_legacy_cache(master_kv), use_cache=True
                )
                sample_kv = output.past_key_values
                logits = output.logits.squeeze(0)[-1]
                probs = softmax(logits)
                # for subsequent sampling steps, options are subword tokens
                # EOW. EOW sums over all space tokens
                subword_probs = probs[subword_ixs]
                space_probs = probs[space_ixs]
                eow_prob = torch.sum(space_probs, dim=0, keepdim=True)
                probs = torch.cat([subword_probs, eow_prob], dim=0)
                ix = torch.multinomial(probs, 1)[0]
                sample_prob = probs[ix]
                curr_log_prob += torch.log2(sample_prob).item()
                if ix.item() == len(probs)-1:
                    eow = True
                    printDebug("EOW chosen, logprob", torch.log2(eow_prob))
                else:
                    eow = False
                while not eow:
                    ix = torch.tensor([subword_ixs[ix]])
                    printDebug("sample:", ix, "logprob:", torch.log2(sample_prob))
                    output = model(
                        input_ids=ix.unsqueeze(0),
                        past_key_values=DynamicCache.from_legacy_cache(sample_kv), use_cache=True
                    )
                    sample_kv = output.past_key_values
                    logits = output.logits.squeeze(0)[-1]
                    probs = softmax(logits)
                    # for subsequent sampling steps, options are subword tokens
                    # EOW. EOW sums over all space tokens
                    subword_probs = probs[subword_ixs]
                    space_probs = probs[space_ixs]
                    eow_prob = torch.sum(space_probs, dim=0, keepdim=True)
                    probs = torch.cat([subword_probs, eow_prob], dim=0)
                    sampled_ix = torch.multinomial(probs, 1)
                    ix = sampled_ix[0]
                    sample_prob = probs[ix]
                    curr_log_prob += torch.log2(sample_prob).item()
                    if ix.item() == len(probs)-1:
                        eow = True
                        printDebug("EOW chosen, logprob", torch.log2(eow_prob))
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
