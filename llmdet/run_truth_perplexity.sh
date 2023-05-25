# 预训练模型
#export PLM=/root/autodl-tmp/gpt2
#export PLM=/root/autodl-tmp/opt-1.3b
#export PLM=/root/autodl-tmp/unilm-base-cased
#export PLM=bigscience/bloom-560m
#export PLM=google/flan-t5-base
export PLM=/root/autodl-tmp/llama-7b-hf
#export PLM=/root/autodl-tmp/gpt-neo-2.7b
#export PLM=facebook/bart-base
#export PLM=/root/autodl-tmp/vicuna-7b

export model=llama

python truth_perplexity.py \
  --model_name_or_path ${PLM} \
  --test_text_file text/test_text.json \
  --output_file result/truth_ppl/${model}.json
