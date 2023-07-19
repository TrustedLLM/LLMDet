import argparse

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
        "--cache_dir",
        type=str,
        default=None,
        help="Path to directory to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        type=bool,
        default=True,
        help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default="main",
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="text_generation/prompt.json",
        help="Path to prompt dataset for text generation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="The number of samples selected for text generation",
    )
    parser.add_argument(
        "--use_auth_token",
        type=bool,
        default=False,
        help="Will use the token generated when running `transformers-cli login` (necessary to use this script with private models).",
    )
    parser.add_argument(
        "--n_generate_text",
        type=int,
        default=96000,
        help="The numbers of generation text.",
    )

    # Samples Arguments
    parser.add_argument(
        "--n_one_grams",
        type=int,
        default=1000,
        help="The number of selected 1-grams in statistics text.",
    )
    parser.add_argument(
        "--n_two_grams",
        type=int,
        default=1000,
        help="The number of selected 2-grams in statistics text.",
    )
    parser.add_argument(
        "--n_three_grams",
        type=int,
        default=1000,
        help="The number of selected 3-grams in statistics text.",
    )
    parser.add_argument(
        "--n_four_grams",
        type=int,
        default=1000,
        help="The number of selected 4-grams in statistics text.",
    )
    parser.add_argument(
        "--two_grams_top_k",
        type=int,
        default=1000,
        help="The number of Top K samples of the next word in 3-grams.",
    )
    parser.add_argument(
        "--three_grams_top_k",
        type=int,
        default=1000,
        help="The number of Top K samples of the next word in 3-grams.",
    )
    parser.add_argument(
        "--four_grams_top_k",
        type=int,
        default=1000,
        help="The number of Top K samples of the next word in 4-grams.",
    )
    parser.add_argument(
        "--five_grams_top_k",
        type=int,
        default=1000,
        help="The number of Top K samples of the next word in 5-grams.",
    )
    parser.add_argument(
        "--two_grams_probs_file",
        type=str,
        default="two_grams_probs.json",
        help="The file of 2-grams probability.",
    )
    parser.add_argument(
        "--three_grams_probs_file",
        type=str,
        default="three_grams_probs.json",
        help="The file of 3-grams probability.",
    )
    parser.add_argument(
        "--four_grams_probs_file",
        type=str,
        default="four_grams_probs.json",
        help="The file of 4-grams probability.",
    )
    parser.add_argument(
        "--five_grams_probs_file",
        type=str,
        default="five_grams_probs.json",
        help="The file of 5-grams probability.",
    )

    # Proxy Perplexity Arguments
    parser.add_argument(
        "--original_test_file",
        type=str,
        default="test_set.json",
        help="The input testing data file.",
    )

    # Output File Arguments
    parser.add_argument(
        "--generated_text_file",
        type=str,
        default="generated_text.json",
        help="The output generated text file.",
    )
    parser.add_argument(
        "--new_test_file",
        type=str,
        default="new_test_set.json",
        help="The new test set file",
    )
    parser.add_argument(
        "--test_ppl_file",
        type=str,
        default="test_ppl.json",
        help="The result of Proxy Perplexity for test texts.",
    )
    args = parser.parse_args()
    return args


