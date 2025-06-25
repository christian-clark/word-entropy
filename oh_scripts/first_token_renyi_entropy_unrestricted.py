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

import os, sys, torch, transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


ALPHA = 0.5


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


def main():
    stories = generate_stories(sys.argv[1])
    model_variant = sys.argv[2].split("/")[-1]
    os.environ["HF_HOME"] = "/scratch/bo2257/hf_cache"

    tokenizer = AutoTokenizer.from_pretrained(sys.argv[2])
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

    batches = []
    #words = []
    for story in stories:
        tokenizer_output = tokenizer(story)
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask

        if prepend_bos:
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
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids[:ctx_size]).unsqueeze(0).cuda(),
                                                        "attention_mask": torch.tensor(attn[:ctx_size]).unsqueeze(0).cuda()}),
                            torch.tensor(ids[1:ctx_size+1]), start_idx, is_first_batch))
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
        batches.append((transformers.BatchEncoding({"input_ids": torch.tensor(ids).unsqueeze(0).cuda(),
                                                    "attention_mask": torch.tensor(attn).unsqueeze(0).cuda()}),
                        torch.tensor(ids[1:]), start_idx, is_first_batch))
    
    print("word entropy")
    for batch in batches:
        batch_input, output_ids, start_idx, is_first_batch = batch

        with torch.no_grad():
            model_output = model(**batch_input)

        toks = tokenizer.convert_ids_to_tokens(output_ids)
        probs = softmax(model_output.logits.double().squeeze(0))
        entropies = torch.log2(torch.sum(probs**ALPHA, dim=1)) / (1-ALPHA)

        curr_w = ""
        curr_t = ""
        curr_entropy = -1

        if not prepend_bos and is_first_batch and "bne" not in model_variant:
            first_id = batch_input.input_ids.cpu().squeeze(0).tolist()[0]
            first_tok = tokenizer.convert_ids_to_tokens([first_id])
            first_tok = tokenizer.convert_tokens_to_string(first_tok).replace(" ", "")
            print(first_tok, 100.)

        for i in range(start_idx, len(toks)):
            cleaned_tok = tokenizer.convert_tokens_to_string([toks[i]]).replace(" ", "")
            if i == start_idx or toks[i].startswith("Ġ"):
                # print(i, start_idx, toks[i])
                if start_idx != 0 and (not curr_t.startswith("Ġ")) and curr_w != "":
                    print(curr_w, 0.)
                elif curr_w != "":
                    print(curr_w, curr_entropy)
                curr_w = cleaned_tok
                curr_t = toks[i]
                curr_entropy = entropies[i].item()
            else:
                curr_w += cleaned_tok
                curr_t += toks[i]
        print(curr_w, curr_entropy)



if __name__ == "__main__":
    main()
