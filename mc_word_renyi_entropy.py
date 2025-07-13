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
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoXTokenizerFast,
    GPTNeoXForCausalLM,
    DynamicCache
)

torch.manual_seed(43)

DEBUG = False
# arbitrarily chosen as a placeholder index
EOW_IX = 50255

class Window:
    def __init__(self, input_ids, attn_mask, word_final, start_ix):
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.word_final = word_final
        self.start_ix = start_ix

    def __str__(self):
        s = "Window:\n\tInput_ids: {}\n\tAttn_mask: {}\n\tWord_final: {}\n\tStart_ix: {}".format(
            self.input_ids, self.attn_mask, self.word_final, self.start_ix
        )
        return s


def debug(*args, **kwargs):
    if DEBUG:
        print("DEBUG:", *args, **kwargs, file=sys.stderr)


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


def main(args):
    stories = generate_stories(args.sentitems)
    model = args.model
    model_variant = model.split("/")[-1]
    num_samples = args.samples
    samples_per_batch = args.samplesPerBatch
    max_iterations = args.maxIter
    context_size = args.contextSize
    alpha = args.alpha
    debug("alpha:", alpha)

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
    if context_size is not None:
        assert context_size < ctx_size
        ctx_size = context_size
    #ctx_size = 16
    debug("ctx_size:", ctx_size)
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
            # subtract max_iterations so context size isn't exceeded when
            # sampling last token in a window
            if ix-curr_start_ix == ctx_size/2 - max_iterations:
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
                word_final=story_word_final[w_start:w_end],
                start_ix=start_ix
            )
            debug("adding window. start: {}, end: {}".format(w_start, w_end))
            windows.append(window)
        
        story_entropies = list()
        for window in windows:
            ids = window.input_ids
            attn = window.attn_mask
            word_final = window.word_final
            start_ix = window.start_ix

            debug(f"\n==== NEW WINDOW ====")
            debug("ids[:20]", ids[:20])
            debug("attn[:20]", attn[:20])
            debug("word_final[:20]", word_final[:20])
            debug("start_ix", start_ix)

            # number of words in the latter part of the window
            num_words = sum(word_final[start_ix:])
            window_log_probs = torch.empty(size=(num_words, 0))

            # just being lazy with this; wouldn't be hard to allow
            # unequal batches
            assert num_samples % samples_per_batch == 0
            num_batches = num_samples // samples_per_batch
            for batch in range(num_batches):
                debug(f"\n==== NEW BATCH ====")
                batch_log_probs = list()
                # process tokens from the first half-window
                if start_ix > 0:
                    prefix_id = torch.tensor(ids[:start_ix]).repeat(samples_per_batch, 1)
                    prefix_attn = torch.tensor(attn[:start_ix]).repeat(samples_per_batch, 1)
                    debug("prefix id:", prefix_id)
                    prefix_output = model(
                        input_ids=prefix_id,
                        attention_mask=prefix_attn,
                        use_cache=True
                    )
                    # samples fork from prefix_kv
                    prefix_kv = prefix_output.past_key_values

                debug("done with prefix; starting the rest of the window. start_ix: {} len(ids): {}".format(start_ix, len(ids)))
                for i in range(start_ix, len(ids)):
                    next_id = torch.tensor(ids[i:i+1]).repeat(samples_per_batch, 1)
                    debug("\nnext id:", next_id)
                    debug("next token:", tokenizer.convert_ids_to_tokens(next_id))
                    if i == 0:
                        attn_mask = torch.tensor(attn[:1]).repeat(samples_per_batch, 1)
                        output = model(
                            input_ids=next_id,
                            attention_mask=attn_mask,
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

                    curr_log_prob = torch.zeros(samples_per_batch)
                    logits = output.logits.squeeze(dim=1)

                    # GPT2 is not trained to put the initial whitespace
                    # on the token immediately after BOS
                    if ids[i] == bos_id:
                        first_logits = logits[:, subword_ixs]
                        #first_probs = softmax(first_logits)
                        first_probs = softmax(first_logits.double())

                    # all words coming after a word other than BOS will start
                    # with whitespace
                    else:
                        #debug("logits shape:", logits.shape)
                        #temp_sm = softmax(logits)
                        #debug("top10:", temp_sm.topk(10))
                        first_logits = logits[:, space_ixs]
                        #first_probs = softmax(first_logits)
                        first_probs = softmax(first_logits.double())

    #                topk = first_probs.topk(10, dim=1)[1]
    #                if ids[i] == bos_id:
    #                    topk = subword_ixs[topk]
    #                else:
    #                    topk = space_ixs[topk]
    #                debug("topk next tokens (after ix conversion):", topk)

                    ix = torch.multinomial(first_probs, 1)
                    # dim: window
                    sample_prob = first_probs.gather(dim=1, index=ix).squeeze(dim=1)
                    curr_log_prob += torch.log2(sample_prob)
                    # indices should be relative to the whole vocabulary
                    if ids[i] == bos_id:
                        ix = subword_ixs.gather(dim=0, index=ix.squeeze(1))
                    else:
                        ix = space_ixs.gather(dim=0, index=ix.squeeze(1))

                    debug("first sample ix:", ix)
                    debug("first sample logprob:", torch.log2(sample_prob))
                    debug("curr log prob:", curr_log_prob)
                    #debug("sample:", tokenizer.convert_ids_to_tokens(ix), "prob:", sample_prob)

                    output = model(
                        input_ids=ix.unsqueeze(1),
                        past_key_values=DynamicCache.from_legacy_cache(master_kv), use_cache=True
                    )
                    # fork the kv cache from the main thread that processes
                    # the input sequence
                    sample_kv = output.past_key_values
                    logits = output.logits.squeeze(1)
                    #probs = softmax(logits)
                    probs = softmax(logits.double())

                    # for subsequent sampling steps, options are subword tokens
                    # or EOW. EOW sums over all space tokens
                    # dim: sample x subwordV
                    subword_probs = probs[:, subword_ixs]
                    # dim: sample x spaceV
                    space_probs = probs[:, space_ixs]
                    # dim: sample x 1
                    eow_prob = torch.sum(space_probs, dim=1, keepdim=True)
                    probs = torch.cat([subword_probs, eow_prob], dim=1)
                    # TODO convert ix to index among all tokens, not just subwords
                    ix = torch.multinomial(probs, 1)

                    sample_prob = probs.gather(dim=1, index=ix).squeeze(dim=1)
                    curr_log_prob += torch.log2(sample_prob)
                    still_going = (ix.squeeze(1) < len(subword_ixs))

                    is_subword = (ix < len(subword_ixs))
                    is_eow = (ix == len(subword_ixs))
                    ix_filtered = ix * is_subword
                    ix = subword_ixs.gather(dim=0, index=ix_filtered.squeeze(1)).unsqueeze(1)
                    ix_filtered = ix * is_subword
                    eow = torch.full((samples_per_batch, 1), EOW_IX)
                    eow_filler = is_eow * eow
                    ix = ix_filtered + eow_filler

                    debug("next sampled ix:", ix)
                    debug("next sample logprob:", torch.log2(sample_prob))
                    debug("curr log probs:", curr_log_prob)
                    debug("still going:", still_going)
                    debug("log probs x still going:", curr_log_prob * still_going)

                    # two subword tokens have been sampled so far
                    iterations = 2
                    while still_going.sum() > 0 and iterations < max_iterations:
                        debug("Current iteration:", iterations)
                        output = model(
                            input_ids=ix,
                            past_key_values=DynamicCache.from_legacy_cache(sample_kv), use_cache=True
                        )
                        sample_kv = output.past_key_values
                        logits = output.logits.squeeze(1)
                        #probs = softmax(logits)
                        probs = softmax(logits.double())

                        # for subsequent sampling steps, options are subword tokens
                        # or EOW. EOW sums over all space tokens
                        # dim: sample x subwordV
                        subword_probs = probs[:, subword_ixs]
                        # dim: sample x spaceV
                        space_probs = probs[:, space_ixs]
                        # dim: sample x 1
                        eow_prob = torch.sum(space_probs, dim=1, keepdim=True)
                        probs = torch.cat([subword_probs, eow_prob], dim=1)
                        # TODO add tensor tracking the step at which each sample finishes
                        ix = torch.multinomial(probs, 1)
                        sample_prob = probs.gather(dim=1, index=ix).squeeze(dim=1)
                        curr_log_prob += torch.log2(sample_prob)*still_going
                        still_going = (still_going * (ix.squeeze(1) < len(subword_ixs)))

                        is_subword = (ix < len(subword_ixs))
                        is_eow = (ix == len(subword_ixs))
                        ix_filtered = ix * is_subword
                        ix = subword_ixs.gather(dim=0, index=ix_filtered.squeeze(1)).unsqueeze(1)
                        ix_filtered = ix * is_subword
                        eow = torch.full((samples_per_batch, 1), EOW_IX)
                        eow_filler = is_eow * eow
                        ix = ix_filtered + eow_filler

                        debug("next sampled ix:", ix)
                        debug("next sample logprob:", torch.log2(sample_prob))
                        debug("curr log probs:", curr_log_prob)
                        debug("still going:", still_going)
                        debug("log probs x still going:", curr_log_prob * still_going)

                        iterations += 1
                        if iterations == max_iterations:
                            debug("NOTE: max iterations reached")
                    # end loop over multi-token continuations
                    debug("log probs for sampled tokens:", curr_log_prob)
                    batch_log_probs.append(curr_log_prob.tolist())
                    #curr_entropy  = -torch.mean(curr_log_prob, dim=0).item()
                    #window_entropies.append(curr_entropy)
                # end loop over tokens in batch
                debug("final batch log probs[:20]:", batch_log_probs[:20])
                batch_log_probs = torch.tensor(batch_log_probs)
                debug("converted:", batch_log_probs)
                window_log_probs = torch.cat([window_log_probs, batch_log_probs], dim=1)
            # end loop over batches for a window
            # alpha=1 corresponds to shannon entropy, i.e. expected surprisal
            if alpha == 1:
                window_entropies = -torch.mean(window_log_probs, dim=1)
            # MC estimate of Renyi entropy for alpha != 1
            else:
                # raw probabilities to alpha-1 power
                #x = torch.pow(2**window_log_probs, alpha-1)
                x = torch.pow(2**window_log_probs.double(), alpha-1)
                #x = 2 ** (alpha-1)*window_log_probs
                debug("story entropies[:20]:", story_entropies[:20])
                x = torch.sum(x, dim=1) / num_samples
                window_entropies = torch.log2(x) / (1-alpha)
            story_entropies.extend(window_entropies.tolist())
        # end loop over windows in a story
        debug("story entropies[:20]:", story_entropies[:20])

        debug("window ids [:20]:")
        debug(ids[start_ix:][:20])
        debug("window word_final [:20]:")
        debug(word_final[start_ix:][:20])
        print_entropies(story_entropies, story_ids, story_word_final, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sentitems")
    parser.add_argument("model")
    parser.add_argument("--samples", "-s", type=int, default=64)
    parser.add_argument("--samplesPerBatch", "-b", type=int, default=2)
    parser.add_argument("--maxIter", "-m", type=int, default=20)
    parser.add_argument("--contextSize", "-c", type=int, default=None)
    parser.add_argument("--alpha", "-a", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
