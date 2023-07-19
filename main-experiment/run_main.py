import os
import json
from functools import partial
import tqdm
import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
from nltk import ngrams
from collections import Counter
import math
import random
import numpy as np


from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, LlamaForCausalLM
from unilm import UniLMTokenizer, UniLMForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer, T5Tokenizer, T5ForConditionalGeneration

from arguments import parse_args

# Load and return model and tokenzier
def load_model(args):
    # Load corresponding model and tokenizer according to  model name ot path
    if args.is_t5:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    elif args.is_llama or args.is_vicuna:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    elif args.is_unilm:
        tokenizer = UniLMTokenizer.from_pretrained(args.model_name_or_path)
        model = UniLMForConditionalGeneration.from_pretrained(args.model_name_or_path)
    elif args.is_bart:
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
        model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    else:
        device = "cpu"

    return tokenizer, model, device

# Generate text according to prompt and model
def generate(input, args, tokenizer, model, device):
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True,
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))
    model = model.to(device)
    # generation function
    generate = partial(model.generate, **gen_kwargs)

    # restrict the length of input
    if args.prompt_max_length:
        pass
    elif hasattr(model.config, "max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings - args.max_new_tokens
    else:
        args.prompt_max_length = 2048 - args.max_new_tokens

    tokd_input = tokenizer(input, return_tensors="pt", add_special_tokens=True, truncation=True,
                           max_length=args.prompt_max_length).to(device)

    torch.manual_seed(args.generation_seed)
    output = generate(**tokd_input)

    # need to isolate the newly generated tokens
    output = output[:, tokd_input["input_ids"].shape[-1]:]

    decoded_output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return decoded_output_text

def generate_seq2seq(input, args, tokenizer, model, device):
    tokenizer_input = tokenizer(input, return_tensors="pt", add_special_tokens=True,
                           max_length=args.prompt_max_length).to(device)
    tokenizer_input_ids = tokenizer_input["input_ids"]
    model = model.to(device)

    outputs = model.generate(
        inputs=tokenizer_input_ids,
        max_length=args.max_new_tokens,
        min_length=300,
        num_return_sequences=1,
        do_sample=True,
        num_beams=args.n_beams,
        temperature=args.sampling_temp,
    )

    # need to isolate the newly generated tokens
    output = outputs[:, tokenizer_input_ids.shape[-1]:]
    decoded_output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return decoded_output_text

# Used to count the top k n-grams  in a given text set
def count_n_grams(text_set, n, top_k, tokenizer):
    n_grams = []
    for text in text_set:
        text_n_grams = ngrams(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)), n)
        n_grams.extend(text_n_grams)

    # Count and sort(according to value in dict, return key) n-grams，
    n_grams = sorted(Counter(n_grams), reverse=True)

    if len(n_grams) < top_k:
        return n_grams
    else:
        return n_grams[:top_k]

def next_token_probs(n_grams, top_k, model, tokenizer, device):
    grams_keys = {}
    grams_probability = {}
    for i in tqdm.tqdm(range(len(n_grams))):
        prompt = tokenizer.decode(n_grams[i])
        tokd_input = tokenizer.encode(prompt, return_tensors="pt").to(device)

        model.to(device)
        with torch.no_grad():
            outputs = model(tokd_input)
            predictions = outputs[0]

        logits = predictions[0, -1, :]

        # probability
        probs = F.softmax(logits, dim=0)

        values, indices = torch.topk(probs, top_k, -1)
        indices = np.array(indices.cpu()).astype(np.uint16)
        values = np.array(values.cpu()).astype(np.float16)
        grams_keys[tuple(n_grams[i])] = indices
        grams_probability[tuple(n_grams[i])] = values

    return grams_keys, grams_probability

def t5_next_token_probs(n_grams, top_k, model, tokenizer, device):
    grams_keys = {}
    grams_probability = {}
    for i in tqdm.tqdm(range(len(n_grams))):
        prompt = tokenizer.decode(n_grams[i])
        tokd_input = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        decoder_input_ids = torch.tensor([[0]]).to(device)
        model.to(device)
        with torch.no_grad():
            outputs = model(tokd_input, decoder_input_ids=decoder_input_ids)
            predictions = outputs.logits

        logits = predictions[0, -1, :]

        # probability
        probs = F.softmax(logits, dim=0)

        values, indices = torch.topk(probs, top_k, -1)
        indices = np.array(indices.cpu()).astype(np.uint16)
        values = np.array(values.cpu()).astype(np.float16)
        grams_keys[tuple(n_grams[i])] = indices
        grams_probability[tuple(n_grams[i])] = values

    return grams_keys, grams_probability

# construct test text set with original text set and generated test set
def construct_test_text_set(original_text_set, generated_test_set, args):
    test_set = []
    for i in generated_test_set:
        if i != "":
            test_set.append({"text": i, "label": args.model_name})
    test_set.extend(original_text_set)
    random.shuffle(test_set)
    return test_set

# Calculate proxy perplexity according to n-grams probability
def proxy_perplexity(text_tokenize_ids, args, probability_2_grams, probability_3_grams, probability_4_grams, probability_5_grams):
    test_ppl = []
    # Calculate proxy perplexity
    for j in tqdm.tqdm(range(len(text_tokenize_ids))):
        for label, tokenize_ids in text_tokenize_ids[j].items():
            ppl_result = {}
            # ppl = math.log2(probability_2_grams[tokenize_ids[0]][str(tokenize_ids[1])])
            ppl = 0
            number_5_grams = 0
            number_4_grams = 0
            number_3_grams = 0
            number_2_grams = 0
            for i in range(3, len(tokenize_ids)-1):

                # 5-grams proxy perplexity
                if str([tokenize_ids[i - j] for j in range(3, -1, -1)]) in probability_5_grams.keys():
                    if str(tokenize_ids[i + 1]) in probability_5_grams[str([tokenize_ids[i - j] for j in range(3, -1, -1)])].keys():
                        ppl = ppl + math.log2(probability_5_grams[str([tokenize_ids[i - j] for j in range(3, -1, -1)])][str(tokenize_ids[i + 1])])
                    else:
                        top_k = len(probability_5_grams[str([tokenize_ids[i - j] for j in range(3, -1, -1)])].keys())
                        sum_probs = sum(probability_5_grams[str([tokenize_ids[i - j] for j in range(3, -1, -1)])].values())
                        if (1 - sum_probs) > 0:
                            ppl = ppl + math.log2((1 - sum_probs) / (args.vocab_length - top_k))
                    number_5_grams = number_5_grams + 1

                # 4-grams proxy perplexity
                elif str([tokenize_ids[i - j] for j in range(2, -1, -1)]) in probability_4_grams.keys():
                    if str(tokenize_ids[i + 1]) in probability_4_grams[str([tokenize_ids[i - j] for j in range(2, -1, -1)])].keys():
                        ppl = ppl + math.log2(probability_4_grams[str([tokenize_ids[i - j] for j in range(2, -1, -1)])][str(tokenize_ids[i + 1])])
                    else:
                        top_k = len(probability_4_grams[str([tokenize_ids[i - j] for j in range(2, -1, -1)])].keys())
                        sum_probs = sum(probability_4_grams[str([tokenize_ids[i - j] for j in range(2, -1, -1)])].values())
                        if (1 - sum_probs) > 0:
                            ppl = ppl + math.log2((1 - sum_probs) / (args.vocab_length - top_k))
                    number_4_grams = number_4_grams + 1

                # 3-grams proxy perplexity
                elif str([tokenize_ids[i - 1], tokenize_ids[i]]) in probability_3_grams.keys():
                    if str(tokenize_ids[i + 1]) in probability_3_grams[str([tokenize_ids[i - 1], tokenize_ids[i]])].keys():
                        ppl = ppl + math.log2(probability_3_grams[str([tokenize_ids[i - 1], tokenize_ids[i]])][str(tokenize_ids[i + 1])])
                    else:
                        top_k = len(probability_3_grams[str([tokenize_ids[i - 1], tokenize_ids[i]])].keys())
                        sum_probs = sum(probability_3_grams[str([tokenize_ids[i - 1], tokenize_ids[i]])].values())
                        if (1 - sum_probs) > 0:
                            ppl = ppl + math.log2((1 - sum_probs) / (args.vocab_length - top_k))
                    number_3_grams = number_3_grams + 1

                # 2-grams proxy perplexity
                elif str([tokenize_ids[i]]) in probability_2_grams.keys():
                    if str(tokenize_ids[i+1]) in probability_2_grams[str([tokenize_ids[i]])].keys():
                        ppl = ppl + math.log2(probability_2_grams[str([tokenize_ids[i]])][str(tokenize_ids[i+1])])
                    else:
                        top_k = len(probability_2_grams[str([tokenize_ids[i]])].keys())
                        sum_probs = sum(probability_2_grams[str([tokenize_ids[i]])].values())
                        ppl = ppl + math.log2((1-sum_probs) / (args.vocab_length - top_k))
                    number_2_grams = number_2_grams + 1
            ppl = round(ppl / (number_2_grams + number_3_grams + number_4_grams + number_5_grams + 1), 2)
            ppl_result[label] = -ppl
            test_ppl.append(ppl_result)
    return test_ppl

def main(args):
    args.is_t5 = any([(model_type in args.model_name_or_path) for model_type in ["t5"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt", "opt", "bloom"]])
    args.is_gpt = any([(model_type in args.model_name_or_path) for model_type in ["gpt"]])
    args.is_opt = any([(model_type in args.model_name_or_path) for model_type in ["opt"]])
    args.is_unilm = any([(model_type in args.model_name_or_path) for model_type in ["unilm"]])
    args.is_llama = any([(model_type in args.model_name_or_path) for model_type in ["llama"]])
    args.is_vicuna = any([(model_type in args.model_name_or_path) for model_type in ["vicuna"]])
    args.is_bart = any([(model_type in args.model_name_or_path) for model_type in ["bart"]])
    args.is_bloom = any([(model_type in args.model_name_or_path) for model_type in ["bloom"]])

    if args.is_gpt:
        args.model_name = "gpt"
        args.vocab_length = 50257
    elif args.is_opt:
        args.model_name = "opt"
        args.vocab_length = 50265
    elif args.is_unilm:
        args.model_name = "unilm"
        args.vocab_length = 28996
    elif args.is_llama:
        args.model_name = "llama"
        args.vocab_length = 32000
    elif args.is_vicuna:
        args.model_name = "vicuna"
        args.vocab_length = 32000
    elif args.is_bart:
        args.model_name = "bart"
        args.vocab_length = 50265
    elif args.is_t5:
        args.model_name = "t5"
        args.vocab_length = 32128
    elif args.is_bloom:
        args.model_name = "bloom"
        args.vocab_length = 250880

    # if args.prompt_file is not None:
    #     extension = args.prompt_file.split(".")[-1]
    #     assert extension in ["csv", "json", "tsv"], "`prompt_file` should be a csv or a json file."
    #
    # # load prompt set
    # prompt = []
    # with open(args.prompt_file, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         try:
    #             text = json.loads(line)
    #             prompt.append(text)
    #
    #         except Exception as e:
    #             print(e, line)
    # prompt = prompt[args.prompt_set * args.n_generate_text:(args.prompt_set + 1) * args.n_generate_text]

    tokenizer, model, device = load_model(args)

    # generate_fn = partial(generate, args=args, tokenizer=tokenizer, model=model, device=device)

    # out_dir = os.path.join("result", args.model_name, str(args.prompt_set))
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    #
    # save_state = os.path.join(out_dir, args.save_state)
    #
    # # check if there is a saved state
    # if os.path.exists(save_state):
    #     try:
    #         with open(save_state, 'rb') as f:
    #             state = pickle.load(f)
    #             generation_text = state['generation_text']
    #             start_batch = state['state_batch'] + 1
    #
    #     except Exception as e:
    #         print(e)
    #
    # else:
    #     generation_text = []
    #     start_batch = 0
    #
    # for batch in tqdm.tqdm(range(start_batch, args.n_generate_text // args.batch_size)):
    #     input = prompt[batch * args.batch_size:(batch + 1) * args.batch_size]
    #     batch_text = []
    #     for idx in range(len(input)):
    #         # batch_text.append(generate_fn(input[idx]))
    #         generated_text = generate_seq2seq(input[idx], args, tokenizer, model, device)
    #         batch_text.append(generated_text)
    #
    #     # save state after each batch
    #     generation_text.extend(batch_text)
    #     if save_state:
    #         state = {
    #             "generation_text": generation_text,
    #             "state_batch": batch
    #         }
    #         with open(save_state, "wb") as f:
    #             pickle.dump(state, f)
    #
    # generation_text_file = os.path.join(out_dir, args.generated_text_file)
    # with open(generation_text_file, "w", encoding='utf-8') as f:
    #     for text in generation_text:
    #         f.write(json.dumps(text, ensure_ascii=False))
    #         f.write('\n')

    # save final state
    # if save_state:
    #     state = {
    #         'generation_text': generation_text,
    #         'state_batch': args.n_generate_text // args.batch_size,
    #     }
    #     with open(save_state, 'wb') as f:
    #         pickle.dump(state, f)

    statistics_text_set = []
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                text = json.loads(line)
                statistics_text_set.append(text)

            except Exception as e:
                print(e, line)
    # statistics_text_set = generation_text[:int(args.n_generate_text * 0.8)]
    # generation_test_text_set = generation_text[int(args.n_generate_text * 0.8):args.n_generate_text]
    #
    # count the number of n-grams
    one_grams = count_n_grams(statistics_text_set, 1, args.n_one_grams, tokenizer)
    two_grams = count_n_grams(statistics_text_set, 2, args.n_two_grams, tokenizer)
    three_grams = count_n_grams(statistics_text_set, 3, args.n_three_grams, tokenizer)
    # four_grams = count_n_grams(statistics_text_set, 4, args.n_four_grams, tokenizer)

    if args.is_t5:
        # samples the next token probability of n-grams
        two_grams_keys, two_grams_probs = t5_next_token_probs(one_grams, args.two_grams_top_k, model, tokenizer, device)
        three_grams_keys, three_grams_probs = t5_next_token_probs(two_grams, args.three_grams_top_k, model, tokenizer, device)
        four_grams_keys, four_grams_probs = t5_next_token_probs(three_grams, args.four_grams_top_k, model, tokenizer, device)
        # five_grams_probs = t5_next_token_probs(four_grams, args.five_grams_top_k, model, tokenizer, device)
    else:
        # samples the next token probability of n-grams
        two_grams_keys, two_grams_probs = next_token_probs(one_grams, args.two_grams_top_k, model, tokenizer, device)
        three_grams_keys, three_grams_probs = next_token_probs(two_grams, args.three_grams_top_k, model, tokenizer, device)
        four_grams_keys, four_grams_probs = next_token_probs(three_grams, args.four_grams_top_k, model, tokenizer, device)
        # five_grams_probs = next_token_probs(four_grams, args.five_grams_top_k, model, tokenizer, device)

    t5 = [two_grams_keys, two_grams_probs, three_grams_keys, three_grams_probs, four_grams_keys, four_grams_probs]
    np.savez(args.original_test_file, t5=t5)
    # two_grams_probs_file = os.path.join(out_dir, args.two_grams_probs_file)
    # with open(two_grams_probs_file, 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(two_grams_probs, ensure_ascii=False))
    #
    # three_grams_probs_file = os.path.join(out_dir, args.three_grams_probs_file)
    # with open(three_grams_probs_file, 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(three_grams_probs, ensure_ascii=False))
    #
    # four_grams_probs_file = os.path.join(out_dir, args.four_grams_probs_file)
    # with open(four_grams_probs_file, 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(four_grams_probs, ensure_ascii=False))
    #
    # five_grams_probs_file = os.path.join(out_dir, args.five_grams_probs_file)
    # with open(five_grams_probs_file, 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(five_grams_probs, ensure_ascii=False))
    #
    # original_text_set = []
    # with open(args.original_test_file, 'r', encoding="utf-8") as f:
    #     for line in f:
    #         try:
    #             text = json.loads(line)
    #             original_text_set.append(text)
    #         except Exception as e:
    #             print(e, line)
    #
    # test_set = construct_test_text_set(original_text_set, generation_test_text_set, args)
    #
    # new_test_file = os.path.join(out_dir, args.new_test_file)
    # with open(new_test_file, "w", encoding='utf-8') as f:
    #     for text in test_set:
    #         f.write(json.dumps(text, ensure_ascii=False))
    #         f.write('\n')
    #
    # test_text_tokenize_ids = []
    # for i in tqdm.tqdm(range(len(test_set))):
    #     tokenize_ids = {}
    #     if args.is_gpt:
    #         text_tokenize = tokenizer.tokenize(test_set[i]["text"], truncation=True)
    #     else:
    #         text_tokenize = tokenizer.tokenize(test_set[i]["text"])
    #     tokenize_ids[text[i]["label"]] = tokenizer.convert_tokens_to_ids(text_tokenize)
    #     # tokens to token_ids
    #     test_text_tokenize_ids.append(tokenize_ids)
    #
    # test_ppl = proxy_perplexity(test_text_tokenize_ids, args, two_grams_probs, three_grams_probs, four_grams_probs, five_grams_probs)
    #
    # test_ppl_file = os.path.join(out_dir, args.test_ppl_file)
    # with open(test_ppl_file, "w", encoding='utf-8') as f:
    #     for ppl in test_ppl:
    #         f.write(json.dumps(ppl, ensure_ascii=False))
    #         f.write('\n')



if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)

