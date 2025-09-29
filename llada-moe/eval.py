import argparse
import logging
from typing import Any, Dict, List, Optional

import jinja2
import torch
import tqdm
from config import LLaDAConfig
from generate import generate
from lm_eval import evaluator
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.loggers import WandbLogger
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.utils import simple_parse_args_string
from modeling import LLaDAMoEModelLM
from transformers import AutoModel, AutoTokenizer, HqqConfig

logging.basicConfig(level=logging.INFO)
eval_logger = logging.getLogger(__name__)


@register_model("llada_moe_eval")
class LLaDAMoEEvalModel(LM):
    """LLaDA-MoE evaluation wrapper for lm-eval."""

    def __init__(
        self,
        pretrained: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        mask_id: int = 156895,
        device: str = "cuda",
        quantization: Optional[str] = None,
        nbits: int = 4,
        batch_size: int = 1,
        **_: Any,
    ) -> None:
        super().__init__()
        if quantization == "hqq":
            hqq_config = HqqConfig(nbits=nbits)
            self.model = AutoModel.from_pretrained(
                pretrained,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                config=LLaDAConfig.from_pretrained(pretrained),
                device_map=device if device != "cpu" else None,
                quantization_config=hqq_config,
            )
            self.device = self.model.device
        elif quantization is None:
            self.model = LLaDAMoEModelLM.from_pretrained(
                pretrained,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                config=LLaDAConfig.from_pretrained(pretrained),
                device_map="cuda"
            )
            self.device = self.model.device

        else:
            raise NotImplementedError(f"Quantization {quantization} not implemented.")

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True
        )
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.remasking = remasking
        self.mask_id = mask_id
        self.batch_size = batch_size

        if self.device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated(self.device)
            eval_logger.info("Model memory allocated on device: %d".format(allocated))
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. Removing system messages and retrying."
            )
            filtered = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                filtered,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        return chat_templated

    @classmethod
    def create_from_arg_string(
        cls,
        arg_string: str,
        additional_config: Optional[Dict[str, Any]] = None,
    ) -> "LLaDAMoEEvalModel":
        args = simple_parse_args_string(arg_string)
        for key, value in args.items():
            if value == "None":
                args[key] = None
        if additional_config:
            args.update(additional_config)
        return cls(**args)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        pbar = tqdm.tqdm(total=len(requests), desc="Generating responses")
        responses: List[str] = []
        for request in requests:
            args = request.args
            context = args[0]
            raw_kwargs = args[1] if len(args) > 1 else {}
            input_ids = self.tokenizer(context, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.device)
            outputs = generate(
                model=self.model,
                prompt=input_ids,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                cfg_scale=self.cfg_scale,
                remasking=self.remasking,
                mask_id=self.mask_id,
            )
            response = self.tokenizer.batch_decode(
                outputs[:, input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            until_tokens = raw_kwargs.get("until") if raw_kwargs else None
            if until_tokens:
                for stop_string in until_tokens:
                    if not stop_string:
                        continue
                    stop_pos = response.find(stop_string)
                    if stop_pos != -1:
                        response = response[:stop_pos]
                        break
            print(f"response: {response}")
            responses.append(response)
            pbar.update(1)
        pbar.close()
        return responses

    def loglikelihood(self, requests: List[Instance]) -> List[tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="LLaDA-MoE Evaluation")
    parser.add_argument(
        "--model", type=str, default="inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
    )
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluation_tracker = EvaluationTracker(output_path="eval_results")
    wandb_logger = WandbLogger(project="llada-moe-eval", job_type="eval")

    model = LLaDAMoEEvalModel(pretrained=args.model, device=args.device)

    result = evaluator.simple_evaluate(
        model,
        model_args="",
        tasks=[args.task],
        device=args.device,
        evaluation_tracker=evaluation_tracker,
        apply_chat_template=True,
        limit=100,
        log_samples=True,
    )
    evaluation_tracker.save_results_aggregated(
        results=result.get("results", {}),
        samples=result.get("samples", {}),
    )
    wandb_logger.post_init(result)
    wandb_logger.log_eval_result()
    wandb_logger.log_eval_samples(samples=result.get("samples", {}))
    print(result)


if __name__ == "__main__":
    cli_evaluate()
