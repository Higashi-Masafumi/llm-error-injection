import argparse
import json
from functools import partial
from pathlib import Path

import zstandard as zstd
from litdata import optimize
from transformers import AutoTokenizer


# 1. Function to tokenize the text contained within the Slimpajama files
def tokenize_fn(filepath: str, tokenizer: AutoTokenizer):
    with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
        for row in f:
            text = json.loads(row)["text"]
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            yield text_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing SlimPajama zst files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the optimized dataset")
    parser.add_argument("--tokenizer_pretrained", type=str, required=True, help="Pretrained tokenizer name or path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tokenizer_pretrained = args.tokenizer_pretrained

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a partial function for tokenization with the loaded tokenizer
    tokenize_partial = partial(tokenize_fn, tokenizer=AutoTokenizer.from_pretrained(tokenizer_pretrained))
    inputs = [str(p) for p in input_dir.rglob("*.zst")]

    # 2. Optimize the dataset using the tokenize function
    optimize(
        fn=tokenize_partial,
        inputs=inputs,
        output_dir=str(output_dir),
        chunk_size=(2049 * 8012),
    )
