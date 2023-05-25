import numpy as np
from tqdm import tqdm
import math
import json
from transformers import AutoTokenizer
from unilm import UniLMTokenizer
import argparse

def parse_args():
    """Command line argument specification"""
    parser = argparse.ArgumentParser(description="Arguments for Generation Text Detection")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--two_grams_probability_file",
        type=str,
        default="two_grams_probability.json",
        help="The file of 2-grams probability.",
    )
    parser.add_argument(
        "--three_grams_probability_file",
        type=str,
        default="three_grams_probs.json",
        help="The file of 3-grams probability.",
    )
    parser.add_argument(
        "--four_grams_probability_file",
        type=str,
        default="four_grams_probs.json",
        help="The file of 4-grams probability.",
    )
    parser.add_argument(
        "--sample_probability_file",
        type=str,
        default="gpt.npz",
        help="The file of 4-grams probability.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="test_text_3.json",
        help="The result of Proxy Perplexity for test texts.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="gpt_test_ppl.json",
        help="The result of Proxy Perplexity for test texts.",
    )
    args = parser.parse_args()
    return args

# Compress n-grams probabilty
def compress_data(n_grams_probability_file):
    # # Load n-grams probability file
    with open(n_grams_probability_file, 'r', encoding="utf-8") as f:
        grams_probs = json.load(f)

    probability_grams = {}
    two_grams_keys = {}
    two_grams_probs_value = {}
    for i in grams_probs.keys():
        two_grams_keys[tuple(np.uint16(eval(i)))] = np.array(list(grams_probs[i].keys())).astype(np.uint16)
        two_grams_probs_value[tuple(np.uint16(eval(i)))] = np.array(list(grams_probs[i].values())).astype(np.float16)
    probability_grams["keys"] = two_grams_keys
    probability_grams["values"] = two_grams_probs_value


    # probability_2_grams = {}
    # for i in two_grams_probs.keys():
    #     # two_grams_prefix = np.array(eval(i)).astype(np.uint16)   tuple(two_grams_prefix)
    #     two_grams_keys = [np.uint16(k) for k in two_grams_probs[i].keys()]
    #     two_grams_probs_value = [np.float16(v) for v in two_grams_probs[i].values()]
    #     probability_2_grams[i] = dict(zip(two_grams_keys, two_grams_probs_value))


    # Compress 3-grams probability
    # three_grams_prefix = [eval(i) for i in three_grams_probs.keys()]
    # three_grams_prefix = np.array(three_grams_prefix).astype(np.uint16)
    # three_grams_keys = []
    # three_grams_probs_value = []
    # for i in three_grams_probs.keys():
    #     three_grams_keys.append(np.array(list(three_grams_probs[i].keys())).astype(np.uint16))
    #     three_grams_probs_value.append(np.array(list(three_grams_probs[i].values())).astype(np.float16))
    # three_grams_probs_dict = []
    # for keys, values in zip(three_grams_keys, three_grams_probs_value):
    #     three_grams_probs_dict.append({k: v for k, v in zip(keys, values)})
    # probability_3_grams = {tuple(k): v for k, v in zip(three_grams_prefix, three_grams_probs_dict)}

    # probability_3_grams = {}
    # for i in three_grams_probs.keys():
    #     three_grams_prefix = np.array(eval(i)).astype(np.uint16)
    #     three_grams_keys = np.array(list(three_grams_probs[i].keys())).astype(np.uint16)
    #     three_grams_probs_value = np.array(list(three_grams_probs[i].values())).astype(np.float16)
    #     three_grams_probs_dict = {k: v for k, v in zip(three_grams_keys, three_grams_probs_value)}
    #     probability_3_grams[tuple(three_grams_prefix)] = three_grams_probs_dict


    # Compress 4-grams probability
    # four_grams_prefix = [eval(i) for i in four_grams_probs.keys()]
    # four_grams_prefix = np.array(four_grams_prefix).astype(np.uint16)
    # four_grams_keys = []
    # four_grams_probs_value = []
    # for i in four_grams_probs.keys():
    #     four_grams_keys.append(np.array(list(four_grams_probs[i].keys())).astype(np.uint16))
    #     four_grams_probs_value.append(np.array(list(four_grams_probs[i].values())).astype(np.float16))
    # four_grams_probs_dict = []
    # for keys, values in zip(four_grams_keys, four_grams_probs_value):
    #     four_grams_probs_dict.append({k: v for k, v in zip(keys, values)})
    # probability_4_grams = {tuple(k): v for k, v in zip(four_grams_prefix, four_grams_probs_dict)}

    # probability_4_grams = {}
    # for i in four_grams_probs.keys():
    #     four_grams_prefix = np.array(eval(i)).astype(np.uint16)
    #     four_grams_keys = np.array(list(four_grams_probs[i].keys())).astype(np.uint16)
    #     four_grams_probs_value = np.array(list(four_grams_probs[i].values())).astype(np.float16)
    #     four_grams_probs_dict = {k: v for k, v in zip(four_grams_keys, four_grams_probs_value)}
    #     probability_4_grams[tuple(four_grams_prefix)] = four_grams_probs_dict
    #
    # n_grams_probability = [probability_2_grams, probability_3_grams, probability_4_grams]

    # # Save model n-grams probability
    # np.savez(args.sample_probability_file,
    #          two_grams_prefix=n_grams_probability['two_grams_prefix'],
    #          two_grams_keys=n_grams_probability['two_grams_keys'],
    #          two_grams_probs_value=n_grams_probability['two_grams_probability'],
    #          three_grams_prefix=n_grams_probability['three_grams_prefix'],
    #          three_grams_keys=n_grams_probability['three_grams_keys'],
    #          three_grams_probs_value=n_grams_probability['three_grams_probability'],
    #          four_grams_prefix=n_grams_probability['four_grams_prefix'],
    #          four_grams_keys=n_grams_probability['four_grams_keys'],
    #          four_grams_probs_value=n_grams_probability['four_grams_probability'])

    # np.savez(args.sample_probability_file, sample_probs=n_grams_probability)


    return probability_grams

def proxy_perplexity(text_tokenize_ids, n_grams_probability, args):
    # Load n-grams probability
    probability_2_grams_keys = n_grams_probability['sample_probs'][0]["keys"]
    probability_2_grams_values = n_grams_probability['sample_probs'][0]["values"]
    probability_3_grams_keys = n_grams_probability['sample_probs'][1]["keys"]
    probability_3_grams_values = n_grams_probability['sample_probs'][1]["values"]
    probability_4_grams_keys = n_grams_probability['sample_probs'][2]["keys"]
    probability_4_grams_values = n_grams_probability['sample_probs'][2]["values"]

    # Calculate proxy perplexity value for test text set
    test_perplexity = []

    for j in tqdm(range(len(text_tokenize_ids))):
        for label, tokenize_ids in text_tokenize_ids[j].items():
            ppl_result = {}
            ppl = 0
            number_3_grams = 0
            number_4_grams = 0
            number_2_grams = 0
            for i in range(2, len(tokenize_ids) - 1):

                # Calculate the perplexity with 4-grams samples probability
                if tuple([tokenize_ids[i - j] for j in range(2, -1, -1)]) in probability_4_grams_keys.keys():
                    if tokenize_ids[i + 1] in probability_4_grams_keys[tuple([tokenize_ids[i - j] for j in range(2, -1, -1)])]:
                        if probability_4_grams_values[tuple([tokenize_ids[i - j] for j in range(2, -1, -1)])][probability_4_grams_keys[tuple([tokenize_ids[i - j] for j in range(2, -1, -1)])].tolist().index(tokenize_ids[i + 1])] > 0:
                            ppl = ppl + math.log2(probability_4_grams_values[tuple([tokenize_ids[i - j] for j in range(2, -1, -1)])][probability_4_grams_keys[tuple([tokenize_ids[i - j] for j in range(2, -1, -1)])].tolist().index(tokenize_ids[i + 1])])
                    else:
                        top_k = len(probability_4_grams_keys[tuple([tokenize_ids[i - j] for j in range(2, -1, -1)])])
                        sum_probs = sum(probability_4_grams_values[tuple([tokenize_ids[i - j] for j in range(2, -1, -1)])])
                        if (1 - sum_probs) > 0:
                            ppl = ppl + math.log2((1 - sum_probs) / (args.vocab_length - top_k))
                    number_4_grams = number_4_grams + 1

                # Calculate the perplexity with 3-grams samples probability
                elif tuple([tokenize_ids[i - 1], tokenize_ids[i]]) in probability_3_grams_keys.keys():
                    if tokenize_ids[i + 1] in probability_3_grams_keys[tuple([tokenize_ids[i - 1], tokenize_ids[i]])]:
                        if probability_3_grams_values[tuple([tokenize_ids[i - 1], tokenize_ids[i]])][probability_3_grams_keys[tuple([tokenize_ids[i - 1], tokenize_ids[i]])].tolist().index(tokenize_ids[i + 1])] > 0:
                            ppl = ppl + math.log2(probability_3_grams_values[tuple([tokenize_ids[i - 1], tokenize_ids[i]])][probability_3_grams_keys[tuple([tokenize_ids[i - 1], tokenize_ids[i]])].tolist().index(tokenize_ids[i + 1])])
                    else:
                        top_k = len(probability_3_grams_keys[tuple([tokenize_ids[i - 1], tokenize_ids[i]])])
                        sum_probs = sum(probability_3_grams_values[tuple([tokenize_ids[i - 1], tokenize_ids[i]])])
                        if (1 - sum_probs) > 0:
                            ppl = ppl + math.log2((1 - sum_probs) / (args.vocab_length - top_k))
                    number_3_grams = number_3_grams + 1

                # Calculate the perplexity with 2-grams samples probability
                elif tuple([tokenize_ids[i]]) in probability_2_grams_keys.keys():
                    if tokenize_ids[i + 1] in probability_2_grams_keys[tuple([tokenize_ids[i]])]:
                        if probability_2_grams_values[tuple([tokenize_ids[i]])][probability_2_grams_keys[tuple([tokenize_ids[i]])].tolist().index(tokenize_ids[i + 1])] > 0:
                            ppl = ppl + math.log2(probability_2_grams_values[tuple([tokenize_ids[i]])][probability_2_grams_keys[tuple([tokenize_ids[i]])].tolist().index(tokenize_ids[i + 1])])
                    else:
                        top_k = len(probability_2_grams_keys[tuple([tokenize_ids[i]])])
                        sum_probs = sum(probability_2_grams_values[tuple([tokenize_ids[i]])])
                        if (1 - sum_probs) > 0:
                            ppl = ppl + math.log2((1 - sum_probs) / (args.vocab_length - top_k))
                    number_2_grams = number_2_grams + 1

            ppl = round(ppl / (number_2_grams + number_3_grams + number_4_grams + 1), 2)

            ppl_result[label] = -ppl
            test_perplexity.append(ppl_result)

    return test_perplexity

# Proxy perplexity function
def new_proxy_perplexity(text_tokenize_ids, args):
    # Load n-grams probability data
    n_grams_probability = np.load(args.sample_probability_file, allow_pickle=True)

    # 2-grams sample probability
    two_grams_prefix = n_grams_probability['two_grams_prefix']
    two_grams_prefix = [str(i) for i in two_grams_prefix]
    two_grams_keys = n_grams_probability['two_grams_keys']
    # # two_grams_keys = [list(i) for i in two_grams_keys]
    # two_grams_keys = [[str(i) for i in keys] for keys in two_grams_keys]
    two_grams_probability = n_grams_probability['two_grams_probs_value']
    two_grams_probs_dict = []
    for keys, values in zip(two_grams_keys, two_grams_probability):
        two_grams_probs_dict.append({str(k): v for k, v in zip(keys, values)})
    probility_2_grams = {k: v for k, v in zip(two_grams_prefix, two_grams_probs_dict)}

    # 3-grams sample probability
    three_grams_prefix = n_grams_probability['three_grams_prefix']
    three_grams_prefix = [str(i) for i in three_grams_prefix]
    three_grams_keys = n_grams_probability['three_grams_keys']
    # three_grams_keys = [list(i) for i in three_grams_keys]
    # three_grams_keys = [[str(i) for i in keys] for keys in three_grams_keys]
    three_grams_probability = n_grams_probability['three_grams_probs_value']
    three_grams_probs_dict = []
    for keys, values in zip(three_grams_keys, three_grams_probability):
        three_grams_probs_dict.append({str(k): v for k, v in zip(keys, values)})
    probility_3_grams = {k: v for k, v in zip(three_grams_prefix, three_grams_probs_dict)}

    # 4-grams sample probability
    four_grams_prefix = n_grams_probability['four_grams_prefix']
    four_grams_prefix = [str(i) for i in four_grams_prefix]
    four_grams_keys = n_grams_probability['four_grams_keys']
    # four_grams_keys = [list(i) for i in four_grams_keys]
    # four_grams_keys = [[str(i) for i in keys] for keys in four_grams_keys]
    four_grams_probability = n_grams_probability['four_grams_probs_value']
    four_grams_probs_dict = []
    for keys, values in zip(four_grams_keys, four_grams_probability):
        four_grams_probs_dict.append({str(k): v for k, v in zip(keys, values)})
    probility_4_grams = {k: v for k, v in zip(four_grams_prefix, four_grams_probs_dict)}

    # Proxy perplexity value for test text set
    test_ppl = []

    # # Calculate proxy perplexity according to n-grams probability
    # for j in tqdm(range(len(text_tokenize_ids))):
    #     for label, tokenize_ids in text_tokenize_ids[j].items():
    #         ppl_result = {}
    #         ppl = 0
    #         # number_5_grams = 0
    #         number_4_grams = 0
    #         number_3_grams = 0
    #         number_2_grams = 0
    #
    #         for i in range(2, len(tokenize_ids) - 1):
    #
    #             # 4-grams proxy perplexity
    #             if str([tokenize_ids[i - k] for k in range(2, -1, -1)]) in four_grams_prefix:
    #                 prefix_4_grams = str([tokenize_ids[i - k] for k in range(2, -1, -1)])
    #                 key_index = four_grams_prefix.index(prefix_4_grams)
    #                 # if tokenize_ids[i + 1] in four_grams_keys[key_index]:
    #                 # probability_index = four_grams_keys[key_index].index(tokenize_ids[i + 1])
    #                 # # probability_index = np.where(four_grams_keys == tokenize_ids[i + 1])[0][0]
    #                 # probability = four_grams_probability[key_index][probability_index]
    #
    #                 if str(tokenize_ids[i + 1]) in four_grams_probs_dict[key_index].keys():
    #                     probability = four_grams_probs_dict[key_index][str(tokenize_ids[i + 1])]
    #
    #                     # if probability != 0:
    #                     ppl += math.log2(probability)
    #                     number_4_grams += 1
    #                 else:
    #                     # top_k = len(four_grams_keys[key_index])
    #                     # sum_probs = sum(four_grams_probability[key_index])
    #
    #                     top_k = len(four_grams_probs_dict[key_index].keys())
    #                     sum_probs = sum(four_grams_probs_dict[key_index].values())
    #
    #                     if (1 - sum_probs) > 0:
    #                         ppl += math.log2((1 - sum_probs) / (args.vocab_length - top_k))
    #                         number_4_grams += 1
    #
    #             # 3-grams proxy perplexity
    #             elif str([tokenize_ids[i - 1], tokenize_ids[i]]) in three_grams_prefix:
    #                 prefix_3_grams = str([tokenize_ids[i - 1], tokenize_ids[i]])
    #                 key_index = three_grams_prefix.index(prefix_3_grams)
    #
    #                 # if tokenize_ids[i + 1] in three_grams_keys[key_index]:
    #                 #     probability_index = three_grams_keys[key_index].index(tokenize_ids[i + 1])
    #                 #     # probability_index = np.where(three_grams_keys == tokenize_ids[i + 1])[0][0]
    #                 #     probability = three_grams_probability[key_index][probability_index]
    #
    #                 if str(tokenize_ids[i + 1]) in three_grams_probs_dict[key_index].keys():
    #                     probability = three_grams_probs_dict[key_index][str(tokenize_ids[i + 1])]
    #
    #                     # if probability != 0:
    #                     ppl += math.log2(probability)
    #                     number_3_grams += 1
    #                 else:
    #                     # top_k = len(three_grams_keys[key_index])
    #                     # sum_probs = sum(three_grams_probability[key_index])
    #
    #                     top_k = len(three_grams_probs_dict[key_index].keys())
    #                     sum_probs = sum(three_grams_probs_dict[key_index].values())
    #
    #                     if (1 - sum_probs) > 0:
    #                         ppl += math.log2((1 - sum_probs) / (args.vocab_length - top_k))
    #                         number_3_grams += 1
    #
    #             # 2-grams proxy perplexity
    #             elif str([tokenize_ids[i]]) in two_grams_prefix:
    #                 prefix_2_grams = str([tokenize_ids[i]])
    #                 key_index = two_grams_prefix.index(prefix_2_grams)
    #
    #                 # if tokenize_ids[i + 1] in two_grams_keys[key_index]:
    #                 #     probability_index = two_grams_keys[key_index].index(tokenize_ids[i + 1])
    #                 #     # probability_index = np.where(two_grams_keys == tokenize_ids[i + 1])[0][0]
    #                 #     probability = two_grams_probability[key_index][probability_index]
    #
    #                 if str(tokenize_ids[i + 1]) in two_grams_probs_dict[key_index].keys():
    #                     probability = two_grams_probs_dict[key_index][str(tokenize_ids[i + 1])]
    #
    #                     # if probability != 0:
    #                     ppl += math.log2(probability)
    #                     number_2_grams += 1
    #                 else:
    #                     # top_k = len(two_grams_keys[key_index])
    #                     # sum_probs = sum(two_grams_probability[key_index])
    #
    #                     top_k = len(two_grams_probs_dict[key_index].keys())
    #                     sum_probs = sum(two_grams_probs_dict[key_index].values())
    #
    #                     if (1 - sum_probs) > 0:
    #                         ppl += math.log2((1 - sum_probs) / (args.vocab_length - top_k))
    #                         number_2_grams += 1
    #         ppl = round(ppl / (number_2_grams + number_3_grams + number_4_grams + 1), 2)
    #         ppl_result[label] = -ppl
    #         print(-ppl)
    #         test_ppl.append(ppl_result)

    # 代理ppl计算
    for j in tqdm(range(len(text_tokenize_ids))):
        for label, tokenize_ids in text_tokenize_ids[j].items():
            ppl_result = {}
            # ppl = math.log2(probility_2_grams[tokenize_ids[0]][str(tokenize_ids[1])])
            ppl = 0
            number_3_grams = 0
            number_4_grams = 0
            number_2_grams = 0
            for i in range(2, len(tokenize_ids) - 1):

                # 4-grams的代理ppl计算
                if str([tokenize_ids[i - j] for j in range(2, -1, -1)]) in probility_4_grams.keys():
                    if str(tokenize_ids[i + 1]) in probility_4_grams[str([tokenize_ids[i - j] for j in range(2, -1, -1)])].keys():
                        if probility_4_grams[str([tokenize_ids[i - j] for j in range(2, -1, -1)])][str(tokenize_ids[i + 1])] > 0:
                            ppl = ppl + math.log2(probility_4_grams[str([tokenize_ids[i - j] for j in range(2, -1, -1)])][str(tokenize_ids[i + 1])])
                    else:
                        top_k = len(probility_4_grams[str([tokenize_ids[i - j] for j in range(2, -1, -1)])].keys())
                        sum_probs = sum(probility_4_grams[str([tokenize_ids[i - j] for j in range(2, -1, -1)])].values())
                        if (1 - sum_probs) > 0:
                            ppl = ppl + math.log2((1 - sum_probs) / (args.vocab_length - top_k))
                    number_4_grams = number_4_grams + 1

                # 3-grams的代理ppl计算
                elif str([tokenize_ids[i - 1], tokenize_ids[i]]) in probility_3_grams.keys():
                    if str(tokenize_ids[i + 1]) in probility_3_grams[str([tokenize_ids[i - 1], tokenize_ids[i]])].keys():
                        if probility_3_grams[str([tokenize_ids[i - 1], tokenize_ids[i]])][str(tokenize_ids[i + 1])] > 0:
                            ppl = ppl + math.log2(probility_3_grams[str([tokenize_ids[i - 1], tokenize_ids[i]])][str(tokenize_ids[i + 1])])
                    else:
                        top_k = len(probility_3_grams[str([tokenize_ids[i - 1], tokenize_ids[i]])].keys())
                        sum_probs = sum(probility_3_grams[str([tokenize_ids[i - 1], tokenize_ids[i]])].values())
                        if (1 - sum_probs) > 0:
                            ppl = ppl + math.log2((1 - sum_probs) / (args.vocab_length - top_k))
                    number_3_grams = number_3_grams + 1

                elif str([tokenize_ids[i]]) in probility_2_grams.keys():
                    if str(tokenize_ids[i + 1]) in probility_2_grams[str([tokenize_ids[i]])].keys():
                        if probility_2_grams[str([tokenize_ids[i]])][str(tokenize_ids[i + 1])] > 0:
                            ppl = ppl + math.log2(probility_2_grams[str([tokenize_ids[i]])][str(tokenize_ids[i + 1])])
                    else:
                        top_k = len(probility_2_grams[str([tokenize_ids[i]])].keys())
                        sum_probs = sum(probility_2_grams[str([tokenize_ids[i]])].values())
                        ppl = ppl + math.log2((1 - sum_probs) / (args.vocab_length - top_k))
                    number_2_grams = number_2_grams + 1

            # 没有一个全局2grams情况下的算法
            ppl = round(ppl / (number_2_grams + number_3_grams + number_4_grams + 1), 2)

            ppl_result[label] = -ppl
            test_ppl.append(ppl_result)

    return test_ppl

def main(args):
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # print(args)
    args.is_opt_model = any([(model_type in args.model_name_or_path) for model_type in ["opt"]])
    args.is_gpt_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt"]])
    args.is_unilm_model = any([(model_type in args.model_name_or_path) for model_type in ["unilm"]])
    if args.is_opt_model:
        args.model_name = "opt"
        args.vocab_length = 50265
    if args.is_gpt_model:
        args.model_name = "gpt"
        args.vocab_length = 50257
    if args.is_unilm_model:
        args.model_name = "unilm"
        args.vocab_length = 28996

    if args.is_unilm_model:
        tokenizer = UniLMTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    test_text = []
    with open(args.test_file, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line)
                test_text.append(text)
            except Exception as e:
                print(e, line)
    # test_text = test_text[0:100]

    test_text_tokenize_ids = []
    for text in test_text:
        tokenize_ids = {}
        if args.is_opt_model:
            # 对text进行tokenize
            text_tokenize = tokenizer.tokenize(text["text"])
        elif args.is_gpt_model:
            # 对text进行tokenize
            text_tokenize = tokenizer.tokenize(text["text"], truncation=True)
        elif args.is_unilm_model:
            text_tokenize = tokenizer.tokenize(text["text"])
        tokenize_ids[text["label"]] = tokenizer.convert_tokens_to_ids(text_tokenize)
        # tokenize转token_ids
        test_text_tokenize_ids.append(tokenize_ids)

    # Load n-grams probability
    n_grams_probability = np.load(args.sample_probability_file, allow_pickle=True)
    # test_ppl = new_proxy_perplexity(test_text_tokenize_ids, n_grams_probability, args)
    test_ppl = proxy_perplexity(test_text_tokenize_ids, n_grams_probability, args)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for i in range(len(test_ppl)):
            f.write(json.dumps(test_ppl[i], ensure_ascii=False))
            f.write('\n')

if __name__ == "__main__":
    args = parse_args()
    # print(args)
    probability_2_grams = compress_data(args.two_grams_probability_file)
    probability_3_grams = compress_data(args.three_grams_probability_file)
    probability_4_grams = compress_data(args.four_grams_probability_file)

    n_grams_probability = [probability_2_grams, probability_3_grams, probability_4_grams]
    # n_grams_probability = [probability_2_grams]

    np.savez(args.sample_probability_file, sample_probs=n_grams_probability)

    main(args)