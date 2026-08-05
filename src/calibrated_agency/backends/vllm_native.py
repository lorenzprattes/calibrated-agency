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

from calibrated_agency.correction import bayesian_correct, normalize_probabilities, softmax
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
    except ImportError:
        torch = Any  # type: ignore[assignment,misc]
        SamplingParams = Any  # type: ignore[assignment,misc]
        VllmConfig = Any  # type: ignore[assignment,misc]
        BatchUpdate = Any  # type: ignore[assignment,misc]

        class LogitsProcessor:  # type: ignore[no-redef]
            pass

        class MoveDirectionality:  # type: ignore[no-redef]
            SWAP = "swap"


VLLM_EXTRA_KEY = "calibrated_agency"


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
    if len(action_token_ids) < 2:
        raise ValueError("at least two action token IDs are required")
    if set(action_token_ids) != set(prior) or set(action_token_ids) != set(target):
        raise ValueError("action token, prior, and target keys must match")
    if len(set(action_token_ids.values())) != len(action_token_ids):
        raise ValueError("action token IDs must be distinct")
    if any(token_id < 0 for token_id in action_token_ids.values()):
        raise ValueError("action token IDs must be nonnegative")
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be finite and in [0, 1]")
    if apply_at_first_token == bool(trigger_token_ids):
        raise ValueError("choose exactly one of first-token or trigger activation")
    normalized_prior = normalize_probabilities(prior)
    normalized_target = normalize_probabilities(target)
    return {
        VLLM_EXTRA_KEY: {
            "action_token_ids": dict(action_token_ids),
            "log_prior": {action: math.log(value) for action, value in normalized_prior.items()},
            "log_target": {
                action: math.log(value) for action, value in normalized_target.items()
            },
            "beta": beta,
            "trigger_token_ids": list(trigger_token_ids or []),
            "apply_at_first_token": apply_at_first_token,
            "compensate_temperature": compensate_temperature,
        }
    }


@dataclass
class _RequestState:
    prompt_token_ids: list[int]
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
        prompt_token_ids: list[int],
        output_token_ids: list[int],
    ) -> _RequestState | None:
        extra_args = params.extra_args
        if not extra_args or VLLM_EXTRA_KEY not in extra_args:
            return None
        config = extra_args[VLLM_EXTRA_KEY]
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
            temperature=float(params.temperature),
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
        config = extra_args[VLLM_EXTRA_KEY]
        required = {
            "action_token_ids",
            "log_prior",
            "log_target",
            "beta",
            "trigger_token_ids",
            "apply_at_first_token",
            "compensate_temperature",
        }
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError("malformed calibrated-agency vLLM processor configuration")
        action_token_ids = config["action_token_ids"]
        if not isinstance(action_token_ids, dict) or len(action_token_ids) < 2:
            raise ValueError("processor requires at least two action token IDs")
        if set(action_token_ids) != set(config["log_prior"]) or set(action_token_ids) != set(
            config["log_target"]
        ):
            raise ValueError("processor action, prior, and target keys must match")
        if bool(config["apply_at_first_token"]) == bool(config["trigger_token_ids"]):
            raise ValueError("processor requires exactly one activation mode")

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, prompt_token_ids, output_token_ids in batch_update.added:
            state = _RequestState.from_sampling_params(
                params, prompt_token_ids, output_token_ids
            )
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
                state.temperature
                if state.compensate_temperature and state.temperature > 0
                else 1.0
            )
            logits[batch_index].fill_(float("-inf"))
            for action, token_id in state.action_token_ids.items():
                logits[batch_index, token_id] = scale * corrected[action]
        return logits


@dataclass(frozen=True)
class VllmNativeDecision:
    action: str
    token_id: int
    probabilities: Mapping[str, float]
    backend_reported_probabilities: Mapping[str, float] | None


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
        prompt_token_ids, action_token_ids = self.resolve_token_ids(request)
        uniform = {action: 1.0 for action in action_token_ids}
        extra_args = make_vllm_extra_args(
            action_token_ids=action_token_ids,
            prior=uniform,
            target=uniform,
            beta=1.0,
            apply_at_first_token=True,
        )
        token_id, raw_scores, _ = self._generate_one(
            prompt_token_ids,
            action_token_ids,
            extra_args=extra_args,
            seed=request.seed,
            temperature=1.0,
        )
        actions_by_token_id = {value: action for action, value in action_token_ids.items()}
        if token_id not in actions_by_token_id:
            raise RuntimeError("native processor generated a token outside the action set")
        return ScoreResult(
            raw_scores=raw_scores,
            probabilities=softmax(raw_scores),
            token_ids=action_token_ids,
            selected_action=actions_by_token_id[token_id],
            metadata={
                "backend": "vllm_native",
                "model": self.model_id,
                "logprobs_mode": self.logprobs_mode,
            },
        )

    def decide_native(
        self,
        request: ScoreRequest,
        *,
        prior: Mapping[str, float],
        target: Mapping[str, float],
        beta: float,
        temperature: float = 1.0,
    ) -> VllmNativeDecision:
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
        raw = self.score(request)
        corrected = bayesian_correct(raw.raw_scores, prior, target, beta)
        reported_probabilities = (
            softmax(reported_scores)
            if self.logprobs_mode in {"processed_logprobs", "processed_logits"}
            else None
        )
        return VllmNativeDecision(
            action=actions_by_token_id[token_id],
            token_id=token_id,
            probabilities=corrected.probabilities,
            backend_reported_probabilities=reported_probabilities,
        )
