# mypy: disable-error-code=import-not-found
"""Hugging Face Transformers in-process scoring and first-token processing."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from calibrated_agency.correction import bayesian_correct, normalize_probabilities, softmax
from calibrated_agency.schemas import ChatMessage, ScoreRequest, ScoreResult

if TYPE_CHECKING:
    import torch
    from transformers import LogitsProcessor
else:
    try:
        import torch
        from transformers import LogitsProcessor
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "transformers"}:
            raise
        torch = Any  # type: ignore[assignment,misc]

        class LogitsProcessor:  # type: ignore[no-redef]
            pass


@dataclass(frozen=True)
class TransformersDecision:
    action: str
    token_id: int
    raw_scores: Mapping[str, float]
    expected_probabilities: Mapping[str, float]
    sampled_probabilities: Mapping[str, float]


def _neutral_generation_config_kwargs(
    *,
    eos_token_id: int | list[int] | None,
    bos_token_id: int | None,
    temperature: float,
) -> dict[str, Any]:
    """Return an action-only config that does not inherit model-card score transforms."""
    return {
        "max_new_tokens": 1,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_k": None,
        "top_p": None,
        "min_p": None,
        "typical_p": None,
        "epsilon_cutoff": None,
        "eta_cutoff": None,
        "repetition_penalty": 1.0,
        "encoder_repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "diversity_penalty": 0.0,
        "sequence_bias": None,
        "bad_words_ids": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "watermarking_config": None,
        "renormalize_logits": False,
        "pad_token_id": eos_token_id,
        "eos_token_id": eos_token_id,
        "bos_token_id": bos_token_id,
        "use_cache": True,
    }


class FirstTokenBayesianLogitsProcessor(LogitsProcessor):
    """Mask and correct the first generated token after a known prompt length.

    Transformers appends its temperature warper after custom processors. Multiplying the
    corrected logits by that temperature keeps the requested calibrated distribution intact.
    """

    def __init__(
        self,
        *,
        prompt_length: int,
        action_token_ids: Mapping[str, int],
        prior: Mapping[str, float],
        target: Mapping[str, float],
        beta: float,
        sampling_temperature: float = 1.0,
        compensate_temperature: bool = True,
    ) -> None:
        if prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        if set(action_token_ids) != set(prior) or set(action_token_ids) != set(target):
            raise ValueError("action token, prior, and target keys must match")
        if len(action_token_ids) < 2 or len(set(action_token_ids.values())) != len(
            action_token_ids
        ):
            raise ValueError("at least two distinct action token IDs are required")
        if not all(
            type(token_id) is int and token_id >= 0
            for token_id in action_token_ids.values()
        ):
            raise ValueError("action token IDs must be nonnegative integers")
        if not math.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("beta must be finite and in [0, 1]")
        if not math.isfinite(sampling_temperature) or sampling_temperature < 0:
            raise ValueError("sampling temperature must be finite and nonnegative")
        self.prompt_length = prompt_length
        self.action_token_ids = dict(action_token_ids)
        self.log_prior = {
            action: math.log(value)
            for action, value in normalize_probabilities(prior).items()
        }
        self.log_target = {
            action: math.log(value)
            for action, value in normalize_probabilities(target).items()
        }
        self.beta = beta
        self.sampling_temperature = sampling_temperature
        self.compensate_temperature = compensate_temperature
        self.last_raw_scores: dict[str, float] | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] != self.prompt_length:
            return scores
        if scores.shape[0] != 1:
            raise RuntimeError("the first release supports one Transformers decision at a time")
        self.last_raw_scores = {
            action: float(scores[0, token_id].item())
            for action, token_id in self.action_token_ids.items()
        }
        corrected_scores = scores.new_full(scores.shape, float("-inf"))
        scale = (
            self.sampling_temperature
            if self.compensate_temperature and self.sampling_temperature > 0
            else 1.0
        )
        for action, token_id in self.action_token_ids.items():
            raw = scores[:, token_id]
            corrected_scores[:, token_id] = scale * (
                self.beta * (raw - self.log_prior[action]) + self.log_target[action]
            )
        return corrected_scores


class TransformersNativeBackend:
    """Direct next-token scoring with a Transformers causal language model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(model) or not hasattr(model, "generate"):
            raise TypeError("model must be a Transformers-compatible causal language model")
        if not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("tokenizer must provide apply_chat_template")
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id or str(getattr(model, "name_or_path", "unknown"))
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._generation_lock = threading.Lock()

    @staticmethod
    def _messages(
        messages: Sequence[ChatMessage],
        assistant_prefix: str | None,
        candidate_label: str | None,
    ) -> tuple[list[dict[str, str]], bool, bool]:
        rendered = [message.as_dict() for message in messages]
        if assistant_prefix is None:
            if candidate_label is None:
                return rendered, True, False
            rendered.append({"role": "assistant", "content": candidate_label})
            return rendered, False, True
        rendered.append(
            {
                "role": "assistant",
                "content": assistant_prefix + (candidate_label or ""),
            }
        )
        return rendered, False, True

    def _render_ids(
        self,
        messages: Sequence[ChatMessage],
        *,
        assistant_prefix: str | None,
        candidate_label: str | None,
    ) -> list[int]:
        rendered, add_generation_prompt, continue_final_message = self._messages(
            messages, assistant_prefix, candidate_label
        )
        token_ids = self.tokenizer.apply_chat_template(
            rendered,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if not isinstance(token_ids, list) or not all(
            isinstance(value, int) for value in token_ids
        ):
            raise RuntimeError("chat template did not return a flat integer token list")
        return token_ids

    def resolve_token_ids(self, request: ScoreRequest) -> tuple[list[int], dict[str, int]]:
        base_ids = self._render_ids(
            request.messages,
            assistant_prefix=request.assistant_prefix,
            candidate_label=None,
        )
        action_token_ids: dict[str, int] = {}
        for candidate in request.action_space.candidates:
            candidate_ids = self._render_ids(
                request.messages,
                assistant_prefix=request.assistant_prefix,
                candidate_label=candidate.label,
            )
            if (
                candidate_ids[: len(base_ids)] != base_ids
                or len(candidate_ids) != len(base_ids) + 1
            ):
                raise RuntimeError(
                    f"label {candidate.label!r} is not one appended token at the rendered boundary"
                )
            action_token_ids[candidate.action] = candidate_ids[-1]
        if len(set(action_token_ids.values())) != len(action_token_ids):
            raise RuntimeError("candidate labels resolve to duplicate token IDs")
        return base_ids, action_token_ids

    def _input_tensor(self, token_ids: Sequence[int]) -> torch.LongTensor:
        try:
            device = self.model.get_input_embeddings().weight.device
        except AttributeError:
            device = next(self.model.parameters()).device
        return torch.tensor([list(token_ids)], dtype=torch.long, device=device)

    def score(self, request: ScoreRequest) -> ScoreResult:
        base_ids, action_token_ids = self.resolve_token_ids(request)
        input_ids = self._input_tensor(base_ids)
        self.model.eval()
        with torch.inference_mode():
            output = self.model(input_ids=input_ids)
        next_logits = output.logits[0, -1]
        raw_scores = {
            action: float(next_logits[token_id].item())
            for action, token_id in action_token_ids.items()
        }
        return ScoreResult(
            raw_scores=raw_scores,
            probabilities=softmax(raw_scores),
            token_ids=action_token_ids,
            selected_action=None,
            metadata={"backend": "transformers_native", "model": self.model_id},
        )

    def decide_native(
        self,
        request: ScoreRequest,
        *,
        prior: Mapping[str, float],
        target: Mapping[str, float],
        beta: float,
        temperature: float = 1.0,
    ) -> TransformersDecision:
        """Sample one corrected action through ``model.generate`` and the native processor."""
        base_ids, action_token_ids = self.resolve_token_ids(request)
        input_ids = self._input_tensor(base_ids)
        processor = FirstTokenBayesianLogitsProcessor(
            prompt_length=len(base_ids),
            action_token_ids=action_token_ids,
            prior=prior,
            target=target,
            beta=beta,
            sampling_temperature=temperature,
        )
        from transformers import GenerationConfig

        generation_config = GenerationConfig(
            **_neutral_generation_config_kwargs(
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=getattr(self.tokenizer, "bos_token_id", None),
                temperature=temperature,
            )
        )
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "generation_config": generation_config,
            "logits_processor": [processor],
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        cuda_devices = (
            [input_ids.device.index]
            if input_ids.device.type == "cuda" and input_ids.device.index is not None
            else []
        )
        with self._generation_lock, torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(request.seed)
            self.model.eval()
            with torch.inference_mode():
                generated = self.model.generate(**generation_kwargs)
        token_id = int(generated.sequences[0, -1].item())
        actions_by_token_id = {value: action for action, value in action_token_ids.items()}
        if token_id not in actions_by_token_id:
            raise RuntimeError("native processor generated a token outside the action set")
        if processor.last_raw_scores is None:
            raise RuntimeError("Transformers did not invoke the action logits processor")
        corrected = bayesian_correct(processor.last_raw_scores, prior, target, beta)
        first_scores = generated.scores[0][0]
        backend_scores = {
            action: float(first_scores[candidate_id].item())
            for action, candidate_id in action_token_ids.items()
        }
        return TransformersDecision(
            action=actions_by_token_id[token_id],
            token_id=token_id,
            raw_scores=processor.last_raw_scores,
            expected_probabilities=corrected.probabilities,
            sampled_probabilities=softmax(backend_scores),
        )
