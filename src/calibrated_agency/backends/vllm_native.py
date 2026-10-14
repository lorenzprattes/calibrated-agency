# mypy: disable-error-code=import-not-found
"""Paper-style in-process vLLM V1 logits processor.

The module remains importable without the optional vLLM extra so that configuration and
unit tests do not require a GPU installation. vLLM itself imports the processor by FQCN in
its EngineCore process.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from calibrated_agency.backends.tokenization import resolve_action_tokens
from calibrated_agency.correction import bayesian_correct, log_normalize_weights, softmax
from calibrated_agency.schemas import ChatMessage, ScoreRequest, ScoreResult

if TYPE_CHECKING:
    import torch
    from vllm import SamplingParams
    from vllm.config import VllmConfig
    from vllm.v1.sample.logits_processor import (
        BatchUpdate,
        LogitsProcessor,
        MoveDirectionality,
    )
else:
    try:
        import torch
        from vllm import SamplingParams
        from vllm.config import VllmConfig
        from vllm.v1.sample.logits_processor import (
            BatchUpdate,
            LogitsProcessor,
            MoveDirectionality,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "vllm"}:
            raise
        torch = Any  # type: ignore[assignment,misc]
        SamplingParams = Any  # type: ignore[assignment,misc]
        VllmConfig = Any  # type: ignore[assignment,misc]
        BatchUpdate = Any  # type: ignore[assignment,misc]

        class LogitsProcessor:  # type: ignore[no-redef]
            pass

        class MoveDirectionality:  # type: ignore[no-redef]
            SWAP = "swap"


VLLM_EXTRA_KEY = "calibrated_agency"
_PROCESSOR_CONFIG_KEYS = {
    "action_token_ids",
    "log_prior",
    "log_target",
    "beta",
    "trigger_token_ids",
    "apply_at_first_token",
    "compensate_temperature",
}


def _validate_processor_config(config: object) -> dict[str, Any]:
    if not isinstance(config, dict) or set(config) != _PROCESSOR_CONFIG_KEYS:
        raise ValueError("malformed calibrated-agency vLLM processor configuration")
    action_token_ids = config["action_token_ids"]
    if not isinstance(action_token_ids, dict) or len(action_token_ids) < 2:
        raise ValueError("processor requires at least two action token IDs")
    if not all(isinstance(action, str) and action for action in action_token_ids):
        raise ValueError("processor action names must be nonempty strings")
    token_ids = list(action_token_ids.values())
    if not all(type(token_id) is int and token_id >= 0 for token_id in token_ids):
        raise ValueError("processor action token IDs must be nonnegative integers")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("processor action token IDs must be distinct")

    for field in ("log_prior", "log_target"):
        values = config[field]
        if not isinstance(values, dict) or set(values) != set(action_token_ids):
            raise ValueError(f"processor {field} keys must match actions")
        if not all(
            type(value) in {int, float} and math.isfinite(value) for value in values.values()
        ):
            raise ValueError(f"processor {field} values must be finite numbers")

    beta = config["beta"]
    if type(beta) not in {int, float} or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("processor beta must be finite and in [0, 1]")
    trigger_token_ids = config["trigger_token_ids"]
    if not isinstance(trigger_token_ids, list) or not all(
        type(token_id) is int and token_id >= 0 for token_id in trigger_token_ids
    ):
        raise ValueError("processor trigger token IDs must be nonnegative integers")
    for field in ("apply_at_first_token", "compensate_temperature"):
        if type(config[field]) is not bool:
            raise ValueError(f"processor {field} must be boolean")
    if config["apply_at_first_token"] == bool(trigger_token_ids):
        raise ValueError("processor requires exactly one activation mode")
    return config


def make_vllm_extra_args(
    *,
    action_token_ids: dict[str, int],
    prior: dict[str, float],
    target: dict[str, float],
    beta: float,
    trigger_token_ids: list[int] | None = None,
    apply_at_first_token: bool = False,
    compensate_temperature: bool = True,
) -> dict[str, Any]:
    """Create validated ``SamplingParams.extra_args`` for the native processor.

    Exactly one activation mode is required: either ``apply_at_first_token=True`` or a
    nonempty trigger. Temperature compensation multiplies corrected logits by the request
    temperature because vLLM applies custom processors before its temperature warper.
    """
    if set(action_token_ids) != set(prior) or set(action_token_ids) != set(target):
        raise ValueError("action token, prior, and target keys must match")
    log_prior = log_normalize_weights(prior)
    log_target = log_normalize_weights(target)
    config = {
        "action_token_ids": dict(action_token_ids),
        "log_prior": log_prior,
        "log_target": log_target,
        "beta": beta,
        "trigger_token_ids": list(trigger_token_ids or []),
        "apply_at_first_token": apply_at_first_token,
        "compensate_temperature": compensate_temperature,
    }
    _validate_processor_config(config)
    return {VLLM_EXTRA_KEY: config}


@dataclass
class _RequestState:
    prompt_token_ids: list[int] | None
    output_token_ids: list[int]
    action_token_ids: dict[str, int]
    log_prior: dict[str, float]
    log_target: dict[str, float]
    beta: float
    trigger_token_ids: list[int]
    apply_at_first_token: bool
    compensate_temperature: bool
    temperature: float
    fired: bool = False

    @classmethod
    def from_sampling_params(
        cls,
        params: SamplingParams,
        prompt_token_ids: list[int] | None,
        output_token_ids: list[int],
    ) -> _RequestState | None:
        extra_args = params.extra_args
        if not extra_args or VLLM_EXTRA_KEY not in extra_args:
            return None
        config = _validate_processor_config(extra_args[VLLM_EXTRA_KEY])
        temperature = float(params.temperature)
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("processor temperature must be finite and nonnegative")
        return cls(
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
            action_token_ids=dict(config["action_token_ids"]),
            log_prior=dict(config["log_prior"]),
            log_target=dict(config["log_target"]),
            beta=float(config["beta"]),
            trigger_token_ids=list(config["trigger_token_ids"]),
            apply_at_first_token=bool(config["apply_at_first_token"]),
            compensate_temperature=bool(config["compensate_temperature"]),
            temperature=temperature,
        )

    def should_apply(self) -> bool:
        if self.fired:
            return False
        if self.apply_at_first_token:
            return not self.output_token_ids
        trigger_length = len(self.trigger_token_ids)
        if trigger_length == 0:
            return False
        if len(self.output_token_ids) >= trigger_length:
            tail = self.output_token_ids[-trigger_length:]
        else:
            if self.prompt_token_ids is None:
                raise RuntimeError("trigger activation requires prompt token IDs")
            sequence = [*self.prompt_token_ids, *self.output_token_ids]
            tail = sequence[-trigger_length:]
        return tail == self.trigger_token_ids


class BayesianActionLogitsProcessor(LogitsProcessor):
    """vLLM V1 batch processor implementing the calibrated action formula."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        del vllm_config, device, is_pin_memory
        self.requests: dict[int, _RequestState] = {}

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        extra_args = params.extra_args
        if not extra_args or VLLM_EXTRA_KEY not in extra_args:
            return
        _validate_processor_config(extra_args[VLLM_EXTRA_KEY])

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, prompt_token_ids, output_token_ids in batch_update.added:
            state = _RequestState.from_sampling_params(params, prompt_token_ids, output_token_ids)
            if state is None:
                self.requests.pop(index, None)
            else:
                self.requests[index] = state
        for source, destination, direction in batch_update.moved:
            source_state = self.requests.pop(source, None)
            destination_state = self.requests.pop(destination, None)
            if source_state is not None:
                self.requests[destination] = source_state
            if destination_state is not None and direction == MoveDirectionality.SWAP:
                self.requests[source] = destination_state

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for batch_index, state in self.requests.items():
            if not state.should_apply():
                continue
            state.fired = True
            raw_scores = {
                action: float(logits[batch_index, token_id].item())
                for action, token_id in state.action_token_ids.items()
            }
            if any(not math.isfinite(value) for value in raw_scores.values()):
                raise RuntimeError("candidate raw logits are nonfinite")
            corrected = {
                action: state.beta * (raw_scores[action] - state.log_prior[action])
                + state.log_target[action]
                for action in state.action_token_ids
            }
            scale = (
                state.temperature if state.compensate_temperature and state.temperature > 0 else 1.0
            )
            if any(not math.isfinite(value * scale) for value in corrected.values()):
                raise RuntimeError("corrected logits exceed floating-point capacity")
            logits[batch_index].fill_(float("-inf"))
            for action, token_id in state.action_token_ids.items():
                logits[batch_index, token_id] = scale * corrected[action]
        return logits


@dataclass(frozen=True)
class VllmNativeDecision:
    action: str
    token_id: int
    raw_scores: Mapping[str, float] | None
    sampled_probabilities: Mapping[str, float]


class VllmNativeBackend:
    """Direct vLLM ``LLM.generate`` adapter for a registered native processor.

    Construct the supplied ``LLM`` with this processor's fully qualified class name and
    declare the engine logprobs mode used in provenance::

        LLM(
            model=...,
            logits_processors=[
                "calibrated_agency.backends.vllm_native:BayesianActionLogitsProcessor"
            ],
            logprobs_mode="raw_logprobs",
        )
    """

    def __init__(
        self,
        llm: Any,
        *,
        model_id: str,
        logprobs_mode: str,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if logprobs_mode not in {
            "raw_logprobs",
            "raw_logits",
            "processed_logprobs",
            "processed_logits",
        }:
            raise ValueError("unknown vLLM logprobs mode")
        if not hasattr(llm, "generate") or not hasattr(llm, "get_tokenizer"):
            raise TypeError("llm must be a vLLM LLM instance")
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        self.model_id = model_id
        self.logprobs_mode = logprobs_mode
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

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
            type(value) is int and value >= 0 for value in token_ids
        ):
            raise RuntimeError("chat template did not return a flat integer token list")
        return token_ids

    def resolve_token_ids(self, request: ScoreRequest) -> tuple[list[int], dict[str, int]]:
        base_ids = self._render_ids(
            request.messages,
            assistant_prefix=request.assistant_prefix,
            candidate_label=None,
        )
        action_token_ids = resolve_action_tokens(
            request.action_space,
            lambda label: self.tokenizer.encode(label, add_special_tokens=False),
            lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False),
        )
        return base_ids, action_token_ids

    @staticmethod
    def _extract_logprobs(completion: Any, action_token_ids: Mapping[str, int]) -> dict[str, float]:
        try:
            first_position = completion.logprobs[0]
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("vLLM completion returned no first-token logprobs") from exc
        raw_scores: dict[str, float] = {}
        for action, token_id in action_token_ids.items():
            entry = first_position.get(token_id)
            if entry is None:
                raise RuntimeError(f"vLLM completion omitted token ID {token_id} for {action!r}")
            value = entry.logprob if hasattr(entry, "logprob") else entry
            raw_scores[action] = float(value)
        return raw_scores

    def _generate_one(
        self,
        prompt_token_ids: list[int],
        action_token_ids: Mapping[str, int],
        *,
        extra_args: Mapping[str, Any],
        seed: int,
        temperature: float,
    ) -> tuple[int, dict[str, float], Any]:
        from vllm import SamplingParams as VllmSamplingParams

        params = VllmSamplingParams(
            temperature=temperature,
            max_tokens=1,
            seed=seed,
            logprobs=len(action_token_ids),
            logprob_token_ids=list(action_token_ids.values()),
            extra_args=dict(extra_args),
        )
        outputs = self.llm.generate(
            [{"prompt_token_ids": prompt_token_ids}], params, use_tqdm=False
        )
        completion = outputs[0].outputs[0]
        token_id = int(completion.token_ids[0])
        raw_scores = self._extract_logprobs(completion, action_token_ids)
        return token_id, raw_scores, completion

    def score(self, request: ScoreRequest) -> ScoreResult:
        return self.score_batch([request])[0]

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[ScoreResult]:
        """Score several requests in one vLLM scheduler batch."""
        if not requests:
            raise ValueError("score_batch requires at least one request")
        from vllm import SamplingParams as VllmSamplingParams

        prompts: list[dict[str, list[int]]] = []
        parameters: list[Any] = []
        token_maps: list[dict[str, int]] = []
        for request in requests:
            prompt_token_ids, action_token_ids = self.resolve_token_ids(request)
            uniform = {action: 1.0 for action in action_token_ids}
            extra_args = make_vllm_extra_args(
                action_token_ids=action_token_ids,
                prior=uniform,
                target=uniform,
                beta=1.0,
                apply_at_first_token=True,
            )
            prompts.append({"prompt_token_ids": prompt_token_ids})
            parameters.append(
                VllmSamplingParams(
                    temperature=1.0,
                    max_tokens=1,
                    seed=request.seed,
                    logprobs=len(action_token_ids),
                    logprob_token_ids=list(action_token_ids.values()),
                    extra_args=extra_args,
                )
            )
            token_maps.append(action_token_ids)

        outputs = self.llm.generate(prompts, parameters, use_tqdm=False)
        results: list[ScoreResult] = []
        for output, action_token_ids in zip(outputs, token_maps, strict=True):
            completion = output.outputs[0]
            token_id = int(completion.token_ids[0])
            raw_scores = self._extract_logprobs(completion, action_token_ids)
            actions_by_token_id = {value: action for action, value in action_token_ids.items()}
            if token_id not in actions_by_token_id:
                raise RuntimeError("native processor generated a token outside the action set")
            results.append(
                ScoreResult(
                    raw_scores=raw_scores,
                    probabilities=softmax(raw_scores),
                    token_ids=action_token_ids,
                    selected_action=actions_by_token_id[token_id],
                    metadata={
                        "backend": "vllm_native",
                        "model": self.model_id,
                        "logprobs_mode": self.logprobs_mode,
                        "batch_size": len(requests),
                        "token_resolution": "bare_label_exact_roundtrip_v1",
                    },
                )
            )
        return results

    def decide_native(
        self,
        request: ScoreRequest,
        *,
        prior: Mapping[str, float],
        target: Mapping[str, float],
        beta: float,
        temperature: float = 1.0,
    ) -> VllmNativeDecision:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("categorical decision temperature must be finite and positive")
        prompt_token_ids, action_token_ids = self.resolve_token_ids(request)
        extra_args = make_vllm_extra_args(
            action_token_ids=action_token_ids,
            prior=dict(prior),
            target=dict(target),
            beta=beta,
            apply_at_first_token=True,
            compensate_temperature=True,
        )
        token_id, reported_scores, _ = self._generate_one(
            prompt_token_ids,
            action_token_ids,
            extra_args=extra_args,
            seed=request.seed,
            temperature=temperature,
        )
        actions_by_token_id = {value: action for action, value in action_token_ids.items()}
        if token_id not in actions_by_token_id:
            raise RuntimeError("native processor generated a token outside the action set")
        raw_scores: Mapping[str, float] | None
        sampled_probabilities: Mapping[str, float]
        if self.logprobs_mode in {"processed_logprobs", "processed_logits"}:
            raw_scores = None
            sampled_probabilities = softmax(reported_scores)
        else:
            raw_scores = reported_scores
            sampled_probabilities = bayesian_correct(
                reported_scores, prior, target, beta
            ).probabilities
        return VllmNativeDecision(
            action=actions_by_token_id[token_id],
            token_id=token_id,
            raw_scores=raw_scores,
            sampled_probabilities=sampled_probabilities,
        )
