"""Hydra + lm-eval harness for BD3LM.

This script registers a custom lm-eval model that wraps BD3LM and provides
generation and loglikelihood for evaluation tasks. It then launches lm-eval's
CLI programmatically, with args composed from a Hydra config (configs/blockdlm.yaml).
"""

from __future__ import annotations

import sys

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import simple_parse_args_string

from .bd3lm import BD3LM
from .diffusion import DiffusionLM


def _nucleus_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Applies top-p filtering to a single-step distribution.

    Args:
      probs: Tensor [B, V] probability distribution.
      p: float in (0,1].
    Returns:
      Filtered distribution with mass re-normalized.
    """
    if p >= 1.0:
        return probs
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    mask = (cum_probs <= p).to(sorted_probs.dtype)
    mask[..., 0] = 1
    sorted_probs = sorted_probs * mask
    # scatter back and renormalize
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, sorted_probs)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return filtered


@register_model("bd3lm_eval")
class BD3LMEvalModel(LM):
    """lm-eval adapter for BD3LM with AR-style decoding using SDPA + KV cache."""

    def __init__(
        self,
        pretrained: str,
        tokenizer: str,
        device: str = "cuda",
        max_new_tokens: int = 128,
        top_p: float = 0.9,
        context_size: int = 1024,
        kv_cache: bool = True,
        dtype: str = "bfloat16",
        **_: dict,
    ) -> None:
        super().__init__()
        # Robust device selection: allow None/"auto" → pick available device
        if device is None or (isinstance(device, str) and device.lower() == "auto"):
            picked = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            picked = device
        self.device = torch.device(picked)
        self.max_new_tokens = int(max_new_tokens)
        self.top_p = float(top_p)
        self.context_size = int(context_size)
        self.kv_cache = bool(kv_cache) and (self.device.type == "cuda")
        torch_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(dtype, torch.bfloat16)

        # Load model + tokenizer
        self.model = BD3LM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        # Sampling requires SDPA backend: update config, blocks and mask
        self.model.config.attn_backend = "sdpa"
        for blk in self.model.backbone.blocks:
            if hasattr(blk, "attn_backend"):
                blk.attn_backend = "sdpa"
        self.model.backbone.gen_mask(
            self.model.config.model_length, self.model.config.block_size, attn_backend="sdpa"
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        # Wrap with DiffusionLM to reuse block-wise logic and samplers
        self.diff = DiffusionLM(self.model.config, tokenizer=self.tokenizer, model=self.model).to(self.device)
        self.block_size = int(self.model.config.block_size)

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path

    @classmethod
    def create_from_arg_string(
        cls: type["BD3LMEvalModel"], arg_string: str, additional_config: dict | None = None
    ) -> "BD3LMEvalModel":
        cfg = simple_parse_args_string(arg_string)
        # Normalize "None" to None
        for k, v in list(cfg.items()):
            if isinstance(v, str) and v.lower() == "none":
                cfg[k] = None

        # Strong typing: cast known fields to expected types
        def as_bool(x):
            if isinstance(x, bool):
                return x
            if isinstance(x, str):
                return x.strip().lower() in {"1", "true", "yes", "y"}
            return bool(x)

        def as_int(x):
            return int(x) if x is not None else x

        def as_float(x):
            return float(x) if x is not None else x

        typed = dict(cfg)
        if "max_new_tokens" in typed:
            typed["max_new_tokens"] = as_int(typed["max_new_tokens"])
        if "context_size" in typed:
            typed["context_size"] = as_int(typed["context_size"])
        if "top_p" in typed:
            typed["top_p"] = as_float(typed["top_p"])
        if "kv_cache" in typed:
            typed["kv_cache"] = as_bool(typed["kv_cache"])
        # passthrough tokenizer name if provided
        if "tokenizer" in typed and typed["tokenizer"] is not None:
            typed["tokenizer"] = str(typed["tokenizer"])

        if additional_config:
            typed.update(additional_config)
        return cls(**typed)

    # ------------------------------ helpers ----------------------------------
    def _prefill_cache(self, input_ids: torch.LongTensor) -> None:
        if not self.kv_cache:
            return
        self.model.reset_kv_cache(eval_batch_size=input_ids.shape[0])
        param_dtype = next(self.model.parameters()).dtype
        zeros = torch.zeros((input_ids.shape[0],), device=self.device, dtype=param_dtype)
        full_len = (input_ids.shape[1] // self.block_size) * self.block_size
        for start in range(0, full_len, self.block_size):
            block = input_ids[:, start : start + self.block_size]
            _ = self.diff(block, zeros, sample_mode=True, store_kv=True)

    def _next_logits(self, current: torch.LongTensor) -> torch.Tensor:
        param_dtype = next(self.model.parameters()).dtype
        zeros = torch.zeros((current.shape[0],), device=self.device, dtype=param_dtype)
        if self.kv_cache:
            inp = current[:, -min(self.block_size, current.shape[1]) :]
            logits = self.diff(inp, zeros, sample_mode=True, store_kv=True)
        else:
            # No cache: use sliding context to limit max length
            inp = current[:, -self.context_size :]
            logits = self.diff(inp, zeros, sample_mode=True, store_kv=False)
        # Use last position for next-token distribution
        return logits[:, -1, :]

    # --------------------------- lm-eval methods ------------------------------
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []
        total = len(requests)
        pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)
        for i, req in enumerate(requests):
            args = req.args
            context = args[0]
            raw_kwargs = args[1] if len(args) > 1 else {}
            until_tokens = raw_kwargs.get("until") if raw_kwargs else None

            ctx_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
            self._prefill_cache(ctx_ids)

            tokens = ctx_ids
            for _ in range(self.max_new_tokens):
                last_logits = self._next_logits(tokens)
                probs = F.softmax(last_logits, dim=-1)
                probs = _nucleus_top_p(probs, self.top_p)
                next_id = torch.multinomial(probs, num_samples=1)  # [B, 1]
                tokens = torch.cat([tokens, next_id], dim=1)

                # simple early-stop if EOS produced
                if self.tokenizer.eos_token_id is not None and int(next_id[0, 0]) == int(
                    self.tokenizer.eos_token_id
                ):
                    break

            gen_text = self.tokenizer.batch_decode(tokens[:, ctx_ids.shape[1] :], skip_special_tokens=True)[0]
            if until_tokens:
                for stop in until_tokens:
                    if not stop:
                        continue
                    pos = gen_text.find(stop)
                    if pos != -1:
                        gen_text = gen_text[:pos]
                        break
            print(f"=== Sample {i+1}/{total} ===\n{gen_text}\n")
            results.append(gen_text)
            # emit progress via callback bus
            pbar.update(1)
        pbar.close()
        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        outputs: list[tuple[float, bool]] = []
        total = len(requests)
        for i, req in enumerate(requests):
            context, continuation = req.args
            context = context or ""
            continuation = continuation or ""

            # Handle empty continuation
            if continuation == "":
                outputs.append((0.0, True))
                continue

            ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
            cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
            if len(cont_ids) == 0:
                outputs.append((0.0, True))
                continue
            if len(ctx_ids) == 0:
                bos = (
                    self.tokenizer.bos_token_id
                    if self.tokenizer.bos_token_id is not None
                    else (self.tokenizer.eos_token_id or 0)
                )
                ctx_ids = [int(bos)]

            tokens = torch.tensor([ctx_ids], dtype=torch.long, device=self.device)
            self._prefill_cache(tokens)

            total_logprob = 0.0
            is_greedy = True
            for tid in cont_ids:
                last_logits = self._next_logits(tokens)
                log_probs = F.log_softmax(last_logits, dim=-1)
                total_logprob += float(log_probs[0, int(tid)])
                pred = int(torch.argmax(log_probs, dim=-1)[0].item())
                is_greedy = is_greedy and (pred == int(tid))
                # teacher forcing
                tokens = torch.cat([tokens, torch.tensor([[int(tid)]], device=self.device)], dim=1)

            outputs.append((total_logprob, is_greedy))
        return outputs

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError


def _build_model_args(cfg: DictConfig) -> str:
    # Require explicit fields; fail fast if missing
    pretrained = cfg.models.pretrained_name
    tokenizer = cfg.models.tokenizer_name
    top_p = cfg.top_p
    max_new_tokens = cfg.max_new_tokens
    context_size = cfg.context_size
    kv_cache = cfg.sampling.kv_cache
    device = cfg.device
    dtype = cfg.dtype
    args = {
        "pretrained": pretrained,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "context_size": context_size,
        "kv_cache": kv_cache,
        "device": device,
        "dtype": dtype,
        "tokenizer": tokenizer,
    }
    return ",".join([f"{k}={v}" for k, v in args.items()])


@hydra.main(config_path="../configs", config_name="blockdlm.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    # Compose lm-eval CLI argv from Hydra config
    # Require explicit config values to avoid unintended overrides
    tasks = cfg.tasks
    batch_size = cfg.batch_size
    num_fewshot = cfg.num_fewshot
    output_path = cfg.output_path
    model_args = _build_model_args(cfg)

    argv = [
        "lm_eval",
        "--model",
        "bd3lm_eval",
        "--tasks",
        tasks if isinstance(tasks, str) else ",".join(tasks),
        "--batch_size",
        str(batch_size),
        "--num_fewshot",
        str(num_fewshot),
        "--model_args",
        model_args,
        "--output_path",
        output_path,
    ]
    sys.argv = argv
    # Print hydra config for reproducibility
    print("Hydra config:\n" + OmegaConf.to_yaml(cfg))
    print("lm-eval argv:", " ".join(argv))
    cli_evaluate()


if __name__ == "__main__":
    main()
