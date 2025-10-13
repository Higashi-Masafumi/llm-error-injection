huggingface-cli login
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B data/SlimPajama-627B
python3 scripts/tokenize_optimize_dataset.py \
    --input_dir data/SlimPajama-627B \
    --output_dir data/SlimPajama-627B-optimized \
    --tokenizer_pretrained "meta-llama/Llama-2-7b-chat-hf"
