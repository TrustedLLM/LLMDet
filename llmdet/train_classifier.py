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
import argparse

from unilm import UniLMTokenizer, UniLMForConditionalGeneration
from transformers import AutoTokenizer, LlamaTokenizer, BartTokenizer, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, LlamaForCausalLM, BartForConditionalGeneration, T5ForConditionalGeneration

from detector import perplexity

def parse_args():
    """Command line argument specification"""
    parser = argparse.ArgumentParser(description="Arguments for Generation Text Detection")

    # Generation Arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--n_grams_probs_file",
        type=str,
        default="gpt.npz",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50257,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="test_file.json",
        help="The number of samples selected for text generation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="gpt_ppl.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt",
    )

    args = parser.parse_args()
    return args


def calculate_perplexity(args):
    n_grams = np.load(args.n_grams_probs_file, allow_pickle=True)
    n_grams_probability = n_grams["t5"]
    # Calculate the proxy perplexity
    perplexity_result = []
    if any([(model_type in args.model_name_or_path) for model_type in ["unilm"]]):
        tokenizer = UniLMTokenizer.from_pretrained(args.model_name_or_path)
    elif any([(model_type in args.model_name_or_path) for model_type in ["llama", "vicuna"]]):
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    elif any([(model_type in args.model_name_or_path) for model_type in ["t5"]]):
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    elif any([(model_type in args.model_name_or_path) for model_type in ["bart"]]):
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    test_text = []
    label = []
    with open(args.test_file, 'r', encoding="utf-8") as f:
        for line in f:
                try:
                    text = json.loads(line)
                    test_text.append(text["text"])
                    label.append(text["label"])
                except Exception as e:
                    print(e, line)

    # # test
    # test_text = test_text[0:100]
    # label = label[0:100]

    text_set_token_ids = []
    for i in tqdm.tqdm(range(len(test_text))):
        if any([(model_type in args.model_name_or_path) for model_type in ["gpt"]]):
            text_tokenize = tokenizer.tokenize(test_text[i], truncation=True)
        else:
            text_tokenize = tokenizer.tokenize(test_text[i])
        text_set_token_ids.append(tokenizer.convert_tokens_to_ids(text_tokenize))

    test_perplexity = perplexity(text_set_token_ids, n_grams_probability, args.vocab_size)

    with open(args.output_file, "w", encoding='utf-8') as f:
        for i, j in zip(label, test_perplexity):
            f.write(json.dumps({i: j}, ensure_ascii=False))
            f.write('\n')

if __name__ == "__main__":
    args = parse_args()
    print(args)
    calculate_perplexity(args)