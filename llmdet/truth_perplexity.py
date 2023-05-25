import json
import torch
import tqdm
import argparse
import torch.nn.functional as F

from transformers import AutoTokenizer, LlamaTokenizer,  AutoModelForCausalLM, LlamaForCausalLM
from unilm import UniLMTokenizer, UniLMForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer, T5Tokenizer, T5ForConditionalGeneration

def parse_args():
    """Command line argument specification"""
    parser = argparse.ArgumentParser(description="The Samples Probability of Large Language Models")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="The name of mask filling model.",
    )
    parser.add_argument(
        "--test_text_file",
        type=str,
        default="/root/autodl-tmp/LLMDet/text/test_text.json",
        help="The path of text set.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="gpt_ppl.json",
    )
    args = parser.parse_args()
    return args

def load_model(args):
    """Load and return the model and tokenizer"""

    # Load and return model and tokenzier

    if args.is_decoder_only_model:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    elif args.is_llama_or_vicuna:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    elif args.is_unilm:
        tokenizer = UniLMTokenizer.from_pretrained(args.model_name_or_path)
        model = UniLMForConditionalGeneration.from_pretrained(args.model_name_or_path)
    elif args.is_bart:
        tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
        model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    elif args.is_t5:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return tokenizer, model, device


def calculate_perplexity(text, model, tokenizer, device):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    # If the length of text is longer than 512, need to split the text into smaller chunks
    stride = 512
    n_chunks = (len(input_ids[0]) - 1) // stride + 1
    loss_sum = 0.0
    with torch.no_grad():
        for i in range(n_chunks - 1):
            start_idx = i * stride
            end_idx = min(start_idx + stride, len(input_ids[0]) - 1)
            chunk_input_ids = input_ids[:, start_idx:end_idx]
            chunk_loss = model(chunk_input_ids, labels=chunk_input_ids).loss
            loss_sum += chunk_loss.item()
        # The length of last chunks must larger than 1
        if n_chunks == 1:
            i = 0
        start_idx = i * stride
        end_idx = min(start_idx + stride, len(input_ids[0]) - 1)
        if start_idx < end_idx:
            chunk_input_ids = input_ids[:, start_idx:end_idx]
            chunk_loss = model(chunk_input_ids, labels=chunk_input_ids).loss
            loss_sum += chunk_loss.item()
        else:
            chunk_loss = torch.tensor(0)
            loss_sum += chunk_loss.item()
    loss_mean = loss_sum / n_chunks
    perplexity = torch.exp(torch.tensor(loss_mean)).item()

    return perplexity


def main(args):

    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt", "opt", "bloom"]])
    args.is_unilm = any([(model_type in args.model_name_or_path) for model_type in ["unilm"]])
    args.is_llama_or_vicuna = any([(model_type in args.model_name_or_path) for model_type in ["llama", "vicuna"]])
    args.is_bart = any([(model_type in args.model_name_or_path) for model_type in ["bart"]])
    args.is_t5 = any([(model_type in args.model_name_or_path) for model_type in ["t5"]])

    test_text = []
    with open(args.test_text_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line)
                test_text.append(text)

            except Exception as e:
                print(e, line)

    # load model and tokenizer
    tokenizer, model, device = load_model(args)

    # test
    # test_text = test_text[:100]

    # Calculate the turth perplexity of text
    truth_perplexity = []
    for i in tqdm.tqdm(range(len(test_text))):
        perplexity = {}
        perplexity_value = calculate_perplexity(test_text[i]["text"], model, tokenizer, device)
        perplexity[test_text[i]["label"]] = perplexity_value
        truth_perplexity.append(perplexity)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i in range(len(truth_perplexity)):
            f.write(json.dumps(truth_perplexity[i], ensure_ascii=False))
            f.write('\n')

if __name__ == "__main__":
    args = parse_args()
    # print(args)

    main(args)


