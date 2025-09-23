model_name="dream_eval"
limit=100
quantization="hqq"  # "hqq", "quanto", or None
nbits=2  # 2, 4, 8 only for "hqq"
max_new_tokens=128
weight_and_biases_project="dream-eval"
task="gsm8k"
batch_size=1

export HF_ALLOW_CODE_EVAL=1

accelerate launch eval.py \
    --model $model_name \
    --limit $limit \
    --model_args "max_new_tokens=$max_new_tokens,quantization=$quantization,nbits=$nbits" \
    --wandb_args "project=$weight_and_biases_project" \
    --tasks $task \
    --batch_size $batch_size \
    --confirm_run_unsafe_code

