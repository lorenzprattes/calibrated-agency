from __future__ import annotations

import urllib.request
from collections.abc import Mapping
from typing import Any

import pytest

from calibrated_agency.backends.vllm_api import VllmApiBackend, VllmApiError
from calibrated_agency.schemas import ActionSpace, ChatMessage, ScoreRequest


class FakeVllmBackend(VllmApiBackend):
    def __init__(self, *, malformed_boundary: bool = False) -> None:
        self.requests: list[tuple[str, Mapping[str, Any] | None]] = []
        self.malformed_boundary = malformed_boundary
        super().__init__(
            "http://unused",
            model="test-model",
            logprobs_mode="raw_logprobs",
            chat_template_kwargs={"enable_thinking": False},
        )

    def _request(self, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.requests.append((path, payload))
        if path in {"/tokenize", "/v1/tokenize"}:
            assert payload is not None
            messages = payload["messages"]
            final = messages[-1]
            content = final["content"] if final["role"] == "assistant" else None
            if content == "<action>":
                return {"tokens": [1, 2, 3]}
            if content == "<action>post":
                return {"tokens": [1, 2, 99, 10] if self.malformed_boundary else [1, 2, 3, 10]}
            if content == "<action>comment":
                return {"tokens": [1, 2, 3, 11]}
            raise AssertionError(f"unexpected tokenize content {content!r}")
        if path == "/v1/chat/completions":
            assert payload is not None
            if payload.get("logprobs"):
                return {
                    "choices": [
                        {
                            "message": {"content": "post"},
                            "finish_reason": "length",
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "token_id:10",
                                        "logprob": -2.0,
                                        "top_logprobs": [
                                            {"token": "token_id:10", "logprob": -2.0},
                                            {"token": "token_id:11", "logprob": -3.0},
                                        ],
                                    }
                                ]
                            },
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {"content": "continued"},
                        "finish_reason": "stop",
                    }
                ]
            }
        raise AssertionError(f"unexpected request path {path}")


def request() -> ScoreRequest:
    return ScoreRequest(
        messages=(ChatMessage("user", "Choose an action"),),
        action_space=ActionSpace.from_mapping({"post": "post", "comment": "comment"}),
        assistant_prefix="<action>",
        seed=42,
    )


def test_vllm_api_scores_explicit_boundary_token_ids() -> None:
    backend = FakeVllmBackend()
    result = backend.score(request())
    assert result.token_ids == {"post": 10, "comment": 11}
    assert result.raw_scores == {"post": -2.0, "comment": -3.0}
    assert result.probabilities["post"] == pytest.approx(0.7310585786)
    assert result.selected_action == "post"
    chat_payload = next(
        payload for path, payload in backend.requests if path.endswith("completions")
    )
    assert chat_payload is not None
    assert chat_payload["allowed_token_ids"] == [10, 11]
    assert chat_payload["logprob_token_ids"] == [10, 11]
    assert chat_payload["max_tokens"] == 1
    assert chat_payload["temperature"] == 1.0
    assert chat_payload["top_p"] == 1.0


def test_vllm_api_decides_from_corrected_scores_not_server_sample() -> None:
    backend = FakeVllmBackend()
    result = backend.decide(
        request(),
        prior={"post": 0.5, "comment": 0.5},
        target={"post": 0.01, "comment": 0.99},
        beta=0.0,
        seed=1,
    )
    assert result.score.selected_action == "post"
    assert result.decision.action == "comment"
    assert result.decision.probabilities == pytest.approx(
        {"post": 0.01, "comment": 0.99}
    )


def test_vllm_api_rejects_label_that_is_not_appended_at_boundary() -> None:
    backend = FakeVllmBackend(malformed_boundary=True)
    with pytest.raises(VllmApiError, match="rendered generation boundary"):
        backend.score(request())


def test_vllm_api_wraps_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class InvalidResponse:
        def __enter__(self) -> InvalidResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"not-json"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: InvalidResponse())
    backend = VllmApiBackend(
        "http://unused", model="model", logprobs_mode="raw_logprobs"
    )
    with pytest.raises(VllmApiError, match="malformed JSON"):
        backend._request("/v1/models")


def test_vllm_api_requires_declared_raw_score_mode() -> None:
    with pytest.raises(ValueError, match="explicit"):
        VllmApiBackend(
            "http://unused", model="model", logprobs_mode="processed_logprobs"
        )


def test_vllm_api_continuation_is_a_separate_request() -> None:
    backend = FakeVllmBackend()
    result = backend.continue_from(
        (ChatMessage("user", "Choose"),),
        assistant_prefix="<action>post</action><post>",
        seed=4,
    )
    assert result.text == "continued"
    assert result.finish_reason == "stop"
    path, payload = backend.requests[-1]
    assert path == "/v1/chat/completions"
    assert payload is not None
    assert payload["continue_final_message"] is True
    assert payload["add_generation_prompt"] is False
