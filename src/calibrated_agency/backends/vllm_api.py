"""vLLM 0.26 OpenAI-compatible HTTP adapter.

This is deliberately a vLLM adapter, not a generic OpenAI backend. It uses vLLM-specific
``allowed_token_ids`` and ``logprob_token_ids`` fields.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from calibrated_agency.backends.base import ContinuationResult
from calibrated_agency.backends.tokenization import resolve_action_tokens
from calibrated_agency.correction import decide as core_decide
from calibrated_agency.correction import softmax
from calibrated_agency.schemas import ChatMessage, ScoredDecision, ScoreRequest, ScoreResult

_TOKEN_ID_PATTERN = re.compile(r"token_id:(\d+)")


class VllmApiError(RuntimeError):
    pass


class VllmApiBackend:
    """Score distinct single-token actions through a controlled vLLM deployment."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        timeout: float = 300.0,
        api_key: str = "EMPTY",
        logprobs_mode: str,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if logprobs_mode not in {"raw_logprobs", "raw_logits"}:
            raise ValueError(
                "the API backend requires an explicit raw_logprobs or raw_logits server mode"
            )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.logprobs_mode = logprobs_mode
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.chat_template_kwargs.setdefault("enable_thinking", False)
        if self.chat_template_kwargs["enable_thinking"] is not False:
            raise ValueError("API action-boundary scoring requires enable_thinking=False")
        self.model = model or self._discover_model()

    def _request(self, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise VllmApiError(f"HTTP {exc.code} from {path}: {body[:2000]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise VllmApiError(f"request to {path} failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VllmApiError(f"{path} returned malformed JSON") from exc
        if not isinstance(decoded, dict):
            raise VllmApiError(f"{path} returned a non-object JSON response")
        return decoded

    def _discover_model(self) -> str:
        payload = self._request("/v1/models")
        models = payload.get("data")
        if not isinstance(models, list) or not models:
            raise VllmApiError("endpoint returned no served model")
        first = models[0]
        if not isinstance(first, dict) or not isinstance(first.get("id"), str):
            raise VllmApiError("endpoint returned malformed model metadata")
        return str(first["id"])

    def _tokenize(self, payload: Mapping[str, Any]) -> list[int]:
        errors: list[str] = []
        for path in ("/tokenize", "/v1/tokenize"):
            try:
                response = self._request(path, payload)
                tokens = response.get("tokens", response.get("token_ids"))
                if not isinstance(tokens, list) or not all(
                    type(token) is int and token >= 0 for token in tokens
                ):
                    raise VllmApiError(f"{path} returned no integer token list")
                return tokens
            except VllmApiError as exc:
                errors.append(str(exc))
        raise VllmApiError("tokenization failed: " + " | ".join(errors))

    def _encode_label(self, label: str) -> list[int]:
        return self._tokenize({"model": self.model, "prompt": label, "add_special_tokens": False})

    def _decode_label(self, token_ids: list[int]) -> str:
        response = self._request("/detokenize", {"model": self.model, "tokens": token_ids})
        text = response.get("prompt")
        if not isinstance(text, str):
            raise VllmApiError("detokenization returned no text prompt")
        return text

    def _messages_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        assistant_prefix: str | None,
        candidate_label: str | None,
    ) -> dict[str, Any]:
        rendered_messages = [message.as_dict() for message in messages]
        if assistant_prefix is None:
            if candidate_label is None:
                add_generation_prompt = True
                continue_final_message = False
            else:
                rendered_messages.append({"role": "assistant", "content": candidate_label})
                add_generation_prompt = False
                continue_final_message = True
        else:
            content = assistant_prefix + (candidate_label or "")
            rendered_messages.append({"role": "assistant", "content": content})
            add_generation_prompt = False
            continue_final_message = True
        return {
            "model": self.model,
            "messages": rendered_messages,
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": continue_final_message,
            "chat_template_kwargs": self.chat_template_kwargs,
            "add_special_tokens": False,
        }

    def resolve_token_ids(self, request: ScoreRequest) -> dict[str, int]:
        """Resolve bare labels without retokenizing the fixed prefix."""
        try:
            token_ids = resolve_action_tokens(
                request.action_space, self._encode_label, self._decode_label
            )
        except RuntimeError as exc:
            raise VllmApiError(str(exc)) from exc
        if len(token_ids) > 128:
            raise VllmApiError("vLLM 0.26 supports at most 128 explicit logprob token IDs")
        return token_ids

    @staticmethod
    def _label_from_serialized_token(
        token: object,
        *,
        labels_by_action: Mapping[str, str],
        actions_by_token_id: Mapping[int, str],
    ) -> str | None:
        serialized = str(token)
        match = _TOKEN_ID_PATTERN.fullmatch(serialized.strip())
        if match:
            return actions_by_token_id.get(int(match.group(1)))
        for action, label in labels_by_action.items():
            if serialized == label:
                return action
        return None

    @classmethod
    def _parse_score_response(
        cls,
        response: Mapping[str, Any],
        *,
        labels_by_action: Mapping[str, str],
        token_ids: Mapping[str, int],
    ) -> tuple[str, dict[str, float]]:
        try:
            choice = response["choices"][0]
            positions = choice["logprobs"]["content"]
            if not isinstance(positions, list) or len(positions) != 1:
                raise VllmApiError("score response must contain exactly one token position")
            first = positions[0]
        except (KeyError, IndexError, TypeError) as exc:
            raise VllmApiError("chat response contains no first-token logprobs") from exc
        if not isinstance(first, dict):
            raise VllmApiError("chat response has malformed first-token logprobs")
        top_logprobs = first.get("top_logprobs")
        if not isinstance(top_logprobs, list):
            raise VllmApiError("chat response has malformed candidate alternatives")
        alternatives = [first, *top_logprobs]
        actions_by_token_id = {token_id: action for action, token_id in token_ids.items()}
        raw_scores: dict[str, float] = {}
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                raise VllmApiError("chat response has malformed candidate alternative")
            action = cls._label_from_serialized_token(
                alternative.get("token"),
                labels_by_action=labels_by_action,
                actions_by_token_id=actions_by_token_id,
            )
            if action is not None:
                try:
                    value = alternative["logprob"]
                    if type(value) not in {float, int} or not math.isfinite(value):
                        raise VllmApiError("candidate score must be numeric and finite")
                    if action in raw_scores and raw_scores[action] != value:
                        raise VllmApiError(f"conflicting duplicate score for {action!r}")
                    raw_scores[action] = float(value)
                except (KeyError, TypeError, ValueError) as exc:
                    raise VllmApiError("candidate score is missing or nonnumeric") from exc
        missing = set(token_ids) - set(raw_scores)
        if missing:
            raise VllmApiError(f"chat response omitted candidate scores: {sorted(missing)}")

        sampled_action = cls._label_from_serialized_token(
            first.get("token"),
            labels_by_action=labels_by_action,
            actions_by_token_id=actions_by_token_id,
        )
        if sampled_action is None:
            raise VllmApiError("sampled token is outside the declared action set")
        return sampled_action, {action: raw_scores[action] for action in token_ids}

    def score(self, request: ScoreRequest) -> ScoreResult:
        token_ids = self.resolve_token_ids(request)
        labels_by_action = {
            candidate.action: candidate.label for candidate in request.action_space.candidates
        }
        payload = self._messages_payload(
            request.messages,
            assistant_prefix=request.assistant_prefix,
            candidate_label=None,
        )
        prompt_ids = self._tokenize(payload)
        if not prompt_ids:
            raise VllmApiError("rendered scoring prompt must not be empty")
        payload.update(
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "max_tokens": 1,
                "logprobs": True,
                "logprob_token_ids": list(token_ids.values()),
                "return_tokens_as_token_ids": True,
                "allowed_token_ids": list(token_ids.values()),
                "seed": request.seed,
                "return_token_ids": True,
            }
        )
        started = time.monotonic()
        response = self._request("/v1/chat/completions", payload)
        self._require_prompt_parity(response, prompt_ids)
        latency = time.monotonic() - started
        selected_action, raw_scores = self._parse_score_response(
            response, labels_by_action=labels_by_action, token_ids=token_ids
        )
        return ScoreResult(
            raw_scores=raw_scores,
            probabilities=softmax(raw_scores),
            token_ids=token_ids,
            selected_action=selected_action,
            metadata={
                "backend": "vllm_api",
                "model": self.model,
                "declared_logprobs_mode": self.logprobs_mode,
                "latency_seconds": latency,
                "selected_action_semantics": "uncorrected_server_sample_ignored_by_decide",
                "token_resolution": "bare_label_exact_roundtrip_v1",
                "prompt_token_ids": prompt_ids,
            },
            raw_response=response,
        )

    @staticmethod
    def _require_prompt_parity(response: Mapping[str, Any], prompt_ids: list[int]) -> None:
        received = response.get("prompt_token_ids")
        if (
            not isinstance(received, list)
            or any(type(value) is not int or value < 0 for value in received)
            or received != prompt_ids
        ):
            raise VllmApiError("generation prompt token IDs differ from tokenization preflight")
        usage = response.get("usage")
        if (
            not isinstance(usage, dict)
            or type(usage.get("prompt_tokens")) is not int
            or usage["prompt_tokens"] != len(prompt_ids)
        ):
            raise VllmApiError("generation prompt token count differs from tokenization preflight")

    def decide(
        self,
        request: ScoreRequest,
        *,
        prior: Mapping[str, float],
        target: Mapping[str, float],
        beta: float,
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> ScoredDecision:
        """Score through vLLM, then sample the corrected policy client-side."""
        if seed is not None and rng is not None:
            raise ValueError("provide at most one of seed or rng")
        score = self.score(request)
        decision = core_decide(
            score.raw_scores,
            prior,
            target,
            beta,
            seed=(request.seed if seed is None else seed) if rng is None else None,
            rng=rng,
        )
        return ScoredDecision(score, decision)

    def continue_from(
        self,
        messages: Sequence[ChatMessage],
        *,
        assistant_prefix: str,
        seed: int,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> ContinuationResult:
        if not messages:
            raise ValueError("continuation requires at least one message")
        if not assistant_prefix:
            raise ValueError("assistant_prefix must not be empty")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        payload = self._messages_payload(
            messages, assistant_prefix=assistant_prefix, candidate_label=None
        )
        prompt_ids = self._tokenize(payload)
        payload.update(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
                "return_token_ids": True,
            }
        )
        started = time.monotonic()
        response = self._request("/v1/chat/completions", payload)
        self._require_prompt_parity(response, prompt_ids)
        latency = time.monotonic() - started
        try:
            choice = response["choices"][0]
            text = str(choice["message"].get("content") or "")
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise VllmApiError("chat response contains no continuation") from exc
        return ContinuationResult(
            text=text,
            finish_reason=None if finish_reason is None else str(finish_reason),
            metadata={"backend": "vllm_api", "model": self.model, "latency_seconds": latency},
            raw_response=response,
        )
