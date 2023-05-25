# 预训练模型
#export PLM=/root/autodl-tmp/gpt2
#export PLM=/root/autodl-tmp/opt-1.3b
#export PLM=/root/autodl-tmp/unilm-base-cased
#export PLM=bigscience/bloom-560m
#export PLM=google/flan-t5-base
#export PLM=/root/autodl-tmp/llama-7b-hf
#export PLM=/root/autodl-tmp/gpt-neo-2.7b
#export PLM=facebook/bart-base
#export PLM=/root/autodl-tmp/vicuna-7b
#export PLM=/root/autodl-tmp/gpt2-large
export PLM=/root/autodl-tmp/opt-2.7b

export model=opt

python run_main.py \
  --model_name_or_path ${PLM} --prompt_file text/${model}_statistics_text.json \
  --n_one_grams 30000 --n_two_grams 70000 --n_three_grams 320000 --n_four_grams 500000 \
  --two_grams_top_k 2000 --three_grams_top_k 1000 --four_grams_top_k 100 --five_grams_top_k 100 \
  --original_test_file npz/${model}_3b.npz