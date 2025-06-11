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

import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoXTokenizerFast,
    GPTNeoXForCausalLM,
    DynamicCache
)

NUM_SAMPLES = 3
DEBUG = True


class Batch:
    def __init__(self, input_ids, attn_mask, word_final, start_ix):
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.word_final = word_final
        self.start_ix = start_ix

    def __str__(self):
        s = "Batch:\n\tInput_ids: {}\n\tAttn_mask: {}\n\tWord_final: {}\n\tStart_ix: {}".format(
            self.input_ids, self.attn_mask, self.word_final, self.start_ix
        )
        return s


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
    #ctx_size = model.config.max_position_embeddings
    ctx_size = 8
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
        for i in range(len(story_ids) - 1):
            curr_id = story_ids[i]
            next_id = story_ids[i+1]
            if curr_id == bos_id or next_id in space_ixs:
                story_word_final.append(True)
            else:
                story_word_final.append(False)
        story_word_final.append(True)

        printDebug(story_ids)
        printDebug(story_word_final)

        # split ids into subsequences, each of length <= 1/2 * ctx_size.
        # make sure that splits don't land mid-word
        # consecutive pairs of these subsequences will make up batches.
        # set up this way so context windows have 50% overlap
        half_window_start_ixs = [0]
        curr_start_ix = 0
        curr_end_ix = 0
        for ix, is_final in enumerate(story_word_final):
            if ix-curr_start_ix == ctx_size/2:
                curr_start_ix = curr_end_ix + 1
                half_window_start_ixs.append(curr_start_ix)
            elif is_final:
                curr_end_ix = ix
        half_window_start_ixs.append(len(story_word_final))
        
        print(half_window_start_ixs)

        batches = list()
        for ix in range(len(half_window_start_ixs)-2):
            w_start = half_window_start_ixs[ix]
            w_mid = half_window_start_ixs[ix+1]
            w_end = half_window_start_ixs[ix+2]
            # first batch will use the entire window
            if ix == 0:
                start_ix = 0
            # later batches will use just the second half of the window
            else:
                start_ix = w_mid - w_start
            batch = Batch(
                input_ids=story_ids[w_start:w_end],
                attn_mask=story_attn[w_start:w_end],
                word_final=story_word_final[w_start:w_end],
                start_ix=start_ix
            )
            batches.append(batch)
            print(batch)
        
        story_entropies = list()
        for batch in batches:
            ids = batch.input_ids
            attn = batch.attn_mask
            word_final = batch.word_final
            start_ix = batch.start_ix

            printDebug(f"\n==== NEW BATCH ====")
            printDebug("ids", ids)
            printDebug("attn", attn)
            printDebug("word_final", word_final)
            printDebug("start_ix", start_ix)

            # process tokens from the first half-window
            if start_ix > 0:
                prefix_id = torch.tensor(ids[:start_ix]).unsqueeze(0)
                prefix_attn = torch.tensor(attn[:start_ix]).unsqueeze(0)
                prefix_output = model(
                    input_ids=prefix_id,
                    attention_mask=prefix_attn,
                    use_cache=True
                )
                # samples fork from prefix_kv
                prefix_kv = prefix_output.past_key_values

            batch_log_probs = list()
            for sample_ix in range(NUM_SAMPLES):
                printDebug(f"\n==== SAMPLE {sample_ix+1} ====")
                sample_log_probs = list()

                for i in range(start_ix, len(ids)):
                    print(i)
                    next_id = torch.tensor(ids[i:i+1])
                    printDebug("\nprocessing next token:", tokenizer.convert_ids_to_tokens(next_id))
                    if i == 0:
                        output = model(
                            input_ids=next_id,
                            attention_mask=torch.tensor(attn[:1]).unsqueeze(0),
                            use_cache=True
                        )
                    elif i == start_ix:
                        output = model(
                            input_ids=next_id,
                            past_key_values=DynamicCache.from_legacy_cache(prefix_kv), use_cache=True
                        )
                    else:
                        output = model(
                            input_ids=next_id,
                            past_key_values=DynamicCache.from_legacy_cache(master_kv), use_cache=True
                        )
                    master_kv = output.past_key_values
                        
                    if not word_final[i]: continue

                    curr_log_prob = 0
                    logits = output.logits.squeeze(dim=0).squeeze(dim=0)
                    # for the first sampling step, only sample over tokens that
                    # are the beginning of a new word
                    space_logits = logits[space_ixs]
                    space_probs = softmax(space_logits)
                    #sampled_ix = torch.multinomial(space_probs, 1, replacement=True)[0]
                    ix = torch.multinomial(space_probs, 1)[0]
                    sample_prob = space_probs[ix]
                    curr_log_prob += torch.log2(sample_prob).item()
                    # indices should be relative to the whole vocabulary
                    ix = torch.tensor([space_ixs[ix]])
                    printDebug("sample:", tokenizer.convert_ids_to_tokens(ix.unsqueeze(0)), "prob:", sample_prob)

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
                        printDebug("EOW chosen, prob", eow_prob.item())
                    else:
                        eow = False
                    while not eow:
                        ix = torch.tensor([subword_ixs[ix]])
                        printDebug("sample:", tokenizer.convert_ids_to_tokens(ix.unsqueeze(0)), "prob:", sample_prob)
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
                            printDebug("EOW chosen, prob", eow_prob.item())
                        else:
                            eow = False

                    sample_log_probs.append(curr_log_prob)
                    printDebug("DONE SAMPLING")
                batch_log_probs.append(sample_log_probs)   

            printDebug("batch log probs:")
            printDebug(batch_log_probs)

            # dim: samples x words
            batch_log_probs = torch.tensor(batch_log_probs)
            assert batch_log_probs.shape[0] == NUM_SAMPLES
            batch_entropies = -torch.mean(batch_log_probs, dim=0)
            story_entropies.extend(batch_entropies.tolist())
            printDebug("batch entropies:")
            printDebug(batch_entropies)
            printDebug("batch ids:")
            printDebug(ids[start_ix:])
            printDebug("batch word_final:")
            printDebug(word_final[start_ix:])
        printDebug("story entropies:", story_entropies)
        print_entropies(story_entropies, story_ids, story_word_final, tokenizer)


if __name__ == "__main__":
    main()
