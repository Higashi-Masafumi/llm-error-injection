import argparse
import logging

import jinja2
import torch
import tqdm
from config import LLaDAConfig
from generation_utils import generate
from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.loggers import WandbLogger
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.utils import simple_parse_args_string
from modeling import LLaDAModelLM
from transformers import AutoModel, AutoTokenizer, HqqConfig

logging.basicConfig(level=logging.INFO)
eval_logger = logging.getLogger(__name__)


@register_model("llada")
class LLaDAEvalModel(LM):
    """LLaDA Evaluation Model

    Args:
        LM (LM): Language Model base class
    """

    def __init__(
        self,
        pretrained: str = "GSAI-ML/LLaDA-8B-Instruct",
        steps: int = 128,
        max_length: int = 128,
        block_length: int = 32,
        device: str = "cuda",
        inject_error: bool = False,
        temperature: float = 0.0,
        nbits: int = 4,
        quantization: str | None = None,
    ) -> None:
        super().__init__()
        if quantization == "hqq":
            hqq_config = HqqConfig(nbits=nbits)
            self.model = AutoModel.from_pretrained(
                pretrained,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                config=LLaDAConfig.from_pretrained(pretrained),
                device_map="cuda",
                quantization_config=hqq_config,
            )

        if quantization is None:
            self.model = LLaDAModelLM.from_pretrained(
                pretrained,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                config=LLaDAConfig.from_pretrained(
                    pretrained, inject_error=inject_error
                ),
            )
            self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.steps = steps
        self.max_length = max_length
        self.block_length = block_length
        self.temperature = temperature
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            # Ensure we can pad batched prompts without hitting HF errors.
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # apply_chat_templateをTrueにした場合は必要
    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Method to apply a chat template to a list of chat history between user and model."""
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )

        return chat_templated

    @classmethod
    def create_from_arg_string(
        cls, arg_string: str, additional_config: dict | None = None
    ) -> "LLaDAEvalModel":
        """Create an instance of the model from a string of arguments.

        Args:
            arg_string (str): Argument string.
            additional_config (dict | None, optional): Additional configuration. Defaults to None.
        Returns:
            T: An instance of the model.
        """
        args = simple_parse_args_string(arg_string)
        if additional_config:
            args.update(additional_config)
        return cls(**args)

    def generate_until(self, requests: list[Instance]) -> list[str]:
        pbar = tqdm.tqdm(total=len(requests), desc="Generating responses")
        responses: list[str] = []
        for request in requests:
            args = request.args
            context = args[0]
            raw_kwargs = args[1] if len(args) > 1 else {}
            input_ids = self.tokenizer(context, return_tensors="pt").input_ids
            out = generate(
                model=self.model,
                prompt=input_ids,
                steps=self.steps,
                gen_length=self.max_length,
                block_length=self.block_length,
                temperature=self.temperature,
            )
            response = self.tokenizer.batch_decode(
                out[:, input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            # until tokenで回答を打ち切るように後処理する
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

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests) -> list[float]:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="LLaDA Evaluation")
    parser.add_argument(
        "--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Model name"
    )
    parser.add_argument("--task", type=str, default="gsm8k", help="Task name")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for evaluation"
    )
    args = parser.parse_args()
    config = {
        "temperature": 0.1,
        "inject_error": True,
        "quantization": "hqq",  # "hqq", "quanto", or None
        "nbits": 2,
    }
    evaluation_tracker = EvaluationTracker(output_path="eval_results")
    wandb_logger = WandbLogger(
        project="llada-eval",
        job_type="eval",
        config=config,
    )
    model = LLaDAEvalModel(
        pretrained=args.model,
        device=args.device,
        inject_error=config["inject_error"],
        temperature=config["temperature"],
        quantization=config["quantization"],
        nbits=config["nbits"],
    )
    # errorを入れない場合
    result = evaluator.simple_evaluate(
        model,
        model_args=f"--inject_error {config['inject_error']} --temperature {config['temperature']}",
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
    torch.cuda.is_available()
    main()
