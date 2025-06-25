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


class Batch:
    def __init__(self, encoding, output_ids, start_idx, is_first_batch):
        self.encoding = encoding
        self.output_ids = output_ids
        self.start_idx = start_idx
        self.is_first_batch = is_first_batch


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

    batches = []
    for story in stories:
        tokenizer_output = tokenizer(story)
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask

        # these tokenizers do not append bos_id by default
        if "gpt" in model_variant or "pythia" in model_variant:
            ids = [bos_id] + ids
            attn = [1] + attn

        start_idx = 0
        # track whether a batch is the first in a story
        # first batch begins with BOS token followed by a subword_ix token
        # all other word-initial tokens are from space_ixs
        is_first_batch = True

        # sliding windows with 50% overlap
        # start_idx is for correctly indexing the "later 50%" of sliding windows
        while len(ids) > ctx_size:
            # # for models that explicitly require the first dimension (batch_size)
            # if "gpt-neox" in model_variant or "pythia" in model_variant or "opt" in model_variant:
            #     batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]).unsqueeze(0),
            #                                                 "attention_mask": torch.tensor(attn[:ctx_size]).unsqueeze(0)}),
            #                     torch.tensor(ids[1:ctx_size+1]), start_idx, True))
            # # for other models
            # elif "gpt" in model_variant:
            #     batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]),
            #                                                 "attention_mask": torch.tensor(attn[:ctx_size])}),
            #                     torch.tensor(ids[1:ctx_size+1]), start_idx, True))
            batch_encoding = transformers.BatchEncoding(
                {"input_ids": torch.tensor(ids[:ctx_size]).unsqueeze(0),
                 "attention_mask": torch.tensor(attn[:ctx_size]).unsqueeze(0)}
            )
            batch = Batch(
                encoding=batch_encoding,
                output_ids=torch.tensor(ids[1:ctx_size+1]),
                start_idx=start_idx,
                is_first_batch=is_first_batch
            )
            batches.append(batch)
            ids = ids[int(ctx_size/2):]
            attn = attn[int(ctx_size/2):]
            start_idx = int(ctx_size/2)
            is_first_batch = False

        # # remaining tokens
        # if "gpt-neox" in model_variant or "pythia" in model_variant or "opt" in model_variant:
        #     batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids).unsqueeze(0),
        #                                                 "attention_mask": torch.tensor(attn).unsqueeze(0)}),
        #                     torch.tensor(ids[1:]), start_idx, False))
        # elif "gpt" in model_variant:
        #     batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids),
        #                                                 "attention_mask": torch.tensor(attn)}),
        #                     torch.tensor(ids[1:]), start_idx, False))
        batch_encoding = transformers.BatchEncoding(
            {"input_ids": torch.tensor(ids).unsqueeze(0),
             "attention_mask": torch.tensor(attn).unsqueeze(0)}
        )
        batch = Batch(
            encoding=batch_encoding,
            output_ids=torch.tensor(ids[1:]),
            start_idx=start_idx,
            is_first_batch=is_first_batch
        )
        batches.append(batch)
    
    print("word entropy")
    for batch in batches:
        batch_input = batch.encoding
        output_ids = batch.output_ids
        start_idx = batch.start_idx
        is_first_batch = batch.is_first_batch

        with torch.no_grad():
            model_output = model(**batch_input)

        toks = tokenizer.convert_ids_to_tokens(output_ids)
        # dim: sents x V
        logits = model_output.logits.squeeze(0)
        if is_first_batch:
            # dim: V
            bos_logits = logits[0]
            # first token after BOS is a subword_ix, not a space_ix
            bos_logits = bos_logits[subword_ixs]
            bos_probs = softmax(bos_logits)
            bos_log_probs = torch.log2(bos_probs)
            bos_entropy = torch.sum(-bos_probs*bos_log_probs, dim=0, keepdim=True)

            logits = logits[1:, space_ixs]
            probs = softmax(logits)
            log_probs = torch.log2(probs)
            entropies = torch.sum(-probs*log_probs, dim=1)
            entropies = torch.cat([bos_entropy, entropies], dim=0)
        else:
            # ignore tokens that aren't the beginning of a new word
            logits = logits[:, space_ixs]
            probs = softmax(logits)
            log_probs = torch.log2(probs)
            entropies = torch.sum(-probs*log_probs, dim=1)

        curr_w = ""
        curr_entropy = -1
        for i in range(start_idx, len(toks)):
            cleaned_tok = tokenizer.convert_tokens_to_string([toks[i]]).replace(" ", "")
            if i == start_idx or toks[i].startswith("Ġ"):
                if curr_w != "":
                    print(curr_w, curr_entropy)
                curr_w = cleaned_tok
                curr_entropy = entropies[i].item()
            else:
                curr_w += cleaned_tok
        print(curr_w, curr_entropy)


if __name__ == "__main__":
    main()
