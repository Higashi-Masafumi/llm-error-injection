import jinja2
import torch
import torch.nn.functional as F
from config import DreamConfig
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import simple_parse_args_string
from modeling import DreamModel
from generation_utils import DreamGenerationConfig
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, HqqConfig
from transformers.utils import logging

eval_logger = logging.get_logger("lm_eval")


@register_model("dream_eval")
class DreamEvalModel(LM):
    """Dream Evaluation Model

    Args:
        LM (LM): Language Model base class
    """

    def __init__(
        self,
        pretrained: str = "Dream-org/Dream-v0-Instruct-7B",
        max_new_tokens: int = 128,
        device: str = "cuda",
        classifier_free_guidance_scale: float = 1.0,
        temperature: float = 0.0,
        sampling_eps: float = 1e-3,
        diffusion_steps: int = 32,
        do_sample: bool = True,
        alg: str | None = "entropy",
        alg_temp: float = 0.0,
        quantization: str | None = None,
        nbits: int = 4,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        if quantization == "hqq":
            hqq_config = HqqConfig(nbits=nbits)
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=pretrained,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                config=DreamConfig.from_pretrained(pretrained),
                device_map="cuda",
                quantization_config=hqq_config,
            )
        elif quantization is None:
            self.model = DreamModel.from_pretrained(
                pretrained,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                config=DreamConfig.from_pretrained(pretrained),
            )
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
            self.device = self.model.device
        else:
            raise NotImplementedError(f"Quantization {quantization} not implemented.")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True
        )
        self.max_new_tokens = max_new_tokens
        self.classifier_free_guidance_scale = classifier_free_guidance_scale
        self.temperature = temperature
        self.sampling_eps = sampling_eps
        self.diffusion_steps = diffusion_steps
        self.do_sample = do_sample
        self.alg = alg
        self.alg_temp = alg_temp

    @property
    def tokenizer_name(self):
        return self.tokenizer.name

    @classmethod
    def create_from_arg_string(
        cls: type["DreamEvalModel"],
        arg_string: str,
        additional_config: dict | None = None,
    ) -> "DreamEvalModel":
        config = simple_parse_args_string(arg_string)
        # Noneの文字列をNoneに変換
        for key, value in config.items():
            if value == "None":
                config[key] = None
        if additional_config:
            config.update(additional_config)
        return cls(**config)

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt=True
    ) -> str:
        """Applies the chat template to the chat history.

        Args:
            chat_history (list[dict[str, str]]): The chat history to apply the template to.
            add_generation_prompt (bool, optional): Whether to add a generation prompt. Defaults to True.

        Returns:
            str: The chat template applied to the chat history.
        """
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
        pbar = tqdm(requests, desc="Generating responses", total=len(requests))
        results = []
        for request in requests:
            args = request.args
            context = args[0]
            raw_kwargs = args[1] if len(args) > 1 else {}
            input_ids = self.tokenizer(context, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.device)
            outputs = self.model.diffusion_generate(
                inputs=input_ids,
                max_new_tokens=self.max_new_tokens,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                alg=self.alg,
                alg_temp=self.alg_temp,
            )
            generated_ids = outputs[:, input_ids.shape[1] :]
            text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ]
            until_tokens = raw_kwargs.get("until") if raw_kwargs else None
            if until_tokens:
                for stop_string in until_tokens:
                    if not stop_string:
                        continue
                    stop_pos = text.find(stop_string)
                    if stop_pos != -1:
                        text = text[:stop_pos]
                        break
            results.append(text)
            print(f"Result: {text}")
            pbar.update(1)
        return results

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []

        for request in requests:
            context, continuation = request.args
            context = context or ""
            continuation = continuation or ""

            # Empty continuations have zero log-probability by definition.
            if continuation == "":
                results.append((0.0, True))
                continue

            context_ids = self.tokenizer.encode(context, add_special_tokens=False)
            continuation_ids = self.tokenizer.encode(
                continuation, add_special_tokens=False
            )

            # If tokenization removes everything (e.g. whitespace-only continuation),
            # treat it as zero-probability mass.
            if len(continuation_ids) == 0:
                results.append((0.0, True))
                continue

            # Ensure at least one prefix token so the first continuation token can be scored.
            if len(context_ids) == 0:
                prefix_id = (
                    self.tokenizer.bos_token_id
                    if self.tokenizer.bos_token_id is not None
                    else self.tokenizer.eos_token_id
                )
                if prefix_id is None:
                    # Fall back to padding token or zero if tokenizer lacks BOS/EOS ids.
                    prefix_id = (
                        self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else 0
                    )
                context_ids = [prefix_id]

            input_ids = context_ids + continuation_ids
            input_tensor = torch.tensor(
                [input_ids], dtype=torch.long, device=self.device
            )

            with torch.no_grad():
                outputs = self.model(input_ids=input_tensor)
                logits = outputs.logits

            # Compute log probabilities over the vocabulary.
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = input_tensor[:, 1:]

            gathered = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(
                -1
            )

            continuation_length = len(continuation_ids)
            context_length = len(context_ids)
            start_idx = context_length - 1
            end_idx = start_idx + continuation_length

            continuation_log_probs = gathered[:, start_idx:end_idx]
            total_log_prob = continuation_log_probs.sum(dim=-1).item()

            greedy_tokens = logits[:, :-1, :].argmax(dim=-1)
            continuation_targets = target_ids[:, start_idx:end_idx]
            continuation_greedy = greedy_tokens[:, start_idx:end_idx]
            is_greedy = bool(torch.equal(continuation_targets, continuation_greedy))

            results.append((total_log_prob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError("loglikelihood_rolling is not implemented yet.")


if __name__ == "__main__":
    cli_evaluate()
