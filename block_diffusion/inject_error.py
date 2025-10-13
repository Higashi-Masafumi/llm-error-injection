import torch

from bd3lms.diffusion import Diffusion
from transformers.utils import logging

logger = logging.get_logger(__name__)


def inject_error(sampler: Diffusion, std: float = 0.1) -> Diffusion:
    """Inject Gaussian noise into the per-block get_qkv outputs to perturb KV cache.

    Args:
        sampler: Diffusion module wrapping a HF BD3LM backbone.
        std: Standard deviation for additive Gaussian noise on qkv.

    Returns:
        The same `sampler` instance with patched blocks.
    """
    if sampler.config.algo.backbone != "hf_dit":
        logger.warning("Only hf_dit backbone is supported for error injection.")
        return sampler

    hf_model = sampler.backbone

    for block in hf_model.backbone.blocks:
        # Preserve original get_qkv to avoid recursive calls
        if not hasattr(block, "get_qkv"):
            logger.debug("Block missing get_qkv; skipping.")
            continue
        orig_get_qkv = block.get_qkv

        def get_qkv_with_error(self, x: torch.Tensor, rotary_cos_sin, store_kv: bool = False, orig_get_qkv=orig_get_qkv):
            qkv = orig_get_qkv(x, rotary_cos_sin, store_kv)
            noise = torch.randn_like(qkv) * std
            return qkv + noise

        logger.info("Injecting KV noise into block")
        # Bind the method to the instance
        block.get_qkv = get_qkv_with_error.__get__(block, block.__class__)

    return sampler
