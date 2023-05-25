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

# {gpt neo: 50257, opt: 50265, unilm: 28996, llama vicuna: 32000, bart: 50265, t5: 32128, bloom: 250880}
export model=opt_3b
export size=50265

python train_classifier.py \
  --model_name_or_path ${PLM} --n_grams_probs_file npz/${model}.npz \
  --test_file text/test_text.json  --output_file result/${model}.json \
  --model ${model} --vocab_size ${size}