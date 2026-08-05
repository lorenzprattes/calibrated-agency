"""vLLM 0.26 OpenAI-compatible HTTP adapter.

This is deliberately a vLLM adapter, not a generic OpenAI backend. It uses vLLM-specific
``allowed_token_ids`` and ``logprob_token_ids`` fields.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from calibrated_agency.backends.base import ContinuationResult
from calibrated_agency.correction import softmax
from calibrated_agency.schemas import ChatMessage, ScoreRequest, ScoreResult

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
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.logprobs_mode = logprobs_mode
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
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
                    isinstance(token, int) for token in tokens
                ):
                    raise VllmApiError(f"{path} returned no integer token list")
                return tokens
            except VllmApiError as exc:
                errors.append(str(exc))
        raise VllmApiError("tokenization failed: " + " | ".join(errors))

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
        }

    def resolve_token_ids(self, request: ScoreRequest) -> dict[str, int]:
        """Verify candidates at the fully rendered generation boundary."""
        base_payload = self._messages_payload(
            request.messages,
            assistant_prefix=request.assistant_prefix,
            candidate_label=None,
        )
        base_ids = self._tokenize(base_payload)
        token_ids: dict[str, int] = {}
        for candidate in request.action_space.candidates:
            candidate_payload = self._messages_payload(
                request.messages,
                assistant_prefix=request.assistant_prefix,
                candidate_label=candidate.label,
            )
            candidate_ids = self._tokenize(candidate_payload)
            if (
                candidate_ids[: len(base_ids)] != base_ids
                or len(candidate_ids) != len(base_ids) + 1
            ):
                raise VllmApiError(
                    f"label {candidate.label!r} is not exactly one appended token at the "
                    "rendered generation boundary"
                )
            token_ids[candidate.action] = candidate_ids[-1]
        if len(set(token_ids.values())) != len(token_ids):
            raise VllmApiError("candidate labels resolve to duplicate token IDs")
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
            first = choice["logprobs"]["content"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise VllmApiError("chat response contains no first-token logprobs") from exc
        if not isinstance(first, dict):
            raise VllmApiError("chat response has malformed first-token logprobs")
        alternatives = [first, *(first.get("top_logprobs") or [])]
        actions_by_token_id = {token_id: action for action, token_id in token_ids.items()}
        raw_scores: dict[str, float] = {}
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                continue
            action = cls._label_from_serialized_token(
                alternative.get("token"),
                labels_by_action=labels_by_action,
                actions_by_token_id=actions_by_token_id,
            )
            if action is not None:
                try:
                    raw_scores[action] = float(alternative["logprob"])
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
        return sampled_action, raw_scores

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
        payload.update(
            {
                "temperature": 1.0,
                "max_tokens": 1,
                "logprobs": True,
                "logprob_token_ids": list(token_ids.values()),
                "return_tokens_as_token_ids": True,
                "allowed_token_ids": list(token_ids.values()),
                "seed": request.seed,
            }
        )
        started = time.monotonic()
        response = self._request("/v1/chat/completions", payload)
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
            },
            raw_response=response,
        )

    def continue_from(
        self,
        messages: Sequence[ChatMessage],
        *,
        assistant_prefix: str,
        seed: int,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> ContinuationResult:
        if not assistant_prefix:
            raise ValueError("assistant_prefix must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        payload = self._messages_payload(
            messages, assistant_prefix=assistant_prefix, candidate_label=None
        )
        payload.update(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            }
        )
        started = time.monotonic()
        response = self._request("/v1/chat/completions", payload)
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
