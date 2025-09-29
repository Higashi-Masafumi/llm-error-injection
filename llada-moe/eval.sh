model=llada_moe_eval
limit=100
quantization=None  # "hqq", "quanto", or None
nbits=4  # 2, 4, 8 only for "hqq
max_new_tokens=128
diffusion_steps=128
weight_and_biases_project="llada-moe-eval-step"
task="gsm8k"
batch_size=1
num_fewshot=5  # Number of few-shot examples to use; set to
# 0 for zero-shot
output_dir="eval_outputs"
export HF_ALLOW_CODE_EVAL=1

accelerate launch eval.py \
    --model $model \
    --limit $limit \
    --model_args "max_new_tokens=$max_new_tokens,quantization=$quantization,nbits=$nbits,diffusion_steps=$diffusion_steps" \
    --tasks $task \
    --num_fewshot $num_fewshot \
    --batch_size $batch_size \
    --apply_chat_template \
    --output_path $output_dir \
    --confirm_run_unsafe_code
