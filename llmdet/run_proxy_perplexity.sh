# 预训练模型
export PLM=gpt2

python new_proxy_perplexity.py \
  --model_name_or_path ${PLM} \
  --two_grams_probability_file  n_grams_probability/large_n_grams/gpt/gpt_frequency_2_grams_29961_probs_2000.json \
  --three_grams_probability_file  n_grams_probability/large_n_grams/gpt/gpt_frequency_3_grams_56936_probs_1000.json \
  --four_grams_probability_file n_grams_probability/large_n_grams/gpt/gpt_frequency_4_grams_321601_probs_100.json \
  --test_file test_set/test_text_3.json \
  --output_path gpt2_ppl.json \
  --sample_probability_file gpt.npz
