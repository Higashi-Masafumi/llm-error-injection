model=llada_eval
limit=100
quantization=None
nbits=
max_new_tokens=128
steps=128
batch_size=1
weight_and_biases_project="llada-eval-quantization"
task="gsm8k"
export HF_ALLOW_CODE_EVAL=1

accelerate launch eval.py \
    --model $model \
    --limit $limit \
    --model_args "max_new_tokens=$max_new_tokens,steps=$steps,quantization=$quantization,nbits=$nbits" \
    --wandb_args "project=$weight_and_biases_project" \
    --tasks $task \
    --batch_size $batch_size \
    --apply_chat_template \
    --confirm_run_unsafe_code
