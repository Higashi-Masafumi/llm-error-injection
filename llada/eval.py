import argparse
import logging

import jinja2
import tqdm
from config import LLaDAConfig
from generation_utils import generate
from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from modeling import LLaDAModelLM
from transformers import AutoTokenizer

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
    ) -> None:
        super().__init__()
        self.model = LLaDAModelLM.from_pretrained(
            pretrained_model_name_or_path=pretrained,
            config=LLaDAConfig.from_pretrained(pretrained),
        )
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.steps = steps
        self.max_length = max_length
        self.block_length = block_length
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
            )
            response = self.tokenizer.batch_decode(
                out[:, input_ids.shape[1]:],
                skip_special_tokens=True
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
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Model name")
    parser.add_argument("--task", type=str, default="gsm8k", help="Task name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    args = parser.parse_args()
    evaluation_tracker = EvaluationTracker(
        output_path="eval_results"
    )
    model = LLaDAEvalModel(pretrained=args.model, device=args.device)
    result = evaluator.simple_evaluate(
        model,
        model_args=args.model,
        tasks=[args.task],
        device=args.device,
        evaluation_tracker=evaluation_tracker,
        limit=1,
        apply_chat_template=True,
    )
    evaluation_tracker.save_results_aggregated(
        results=result.pop("samples", {})
    )
    print(result)

if __name__ == "__main__":
    main()
