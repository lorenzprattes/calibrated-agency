from __future__ import annotations

import math
import random
import urllib.request
from collections.abc import Mapping
from typing import Any

import pytest

from calibrated_agency.backends.vllm_api import VllmApiBackend, VllmApiError
from calibrated_agency.schemas import ActionSpace, ChatMessage, ScoreRequest


class FakeVllmBackend(VllmApiBackend):
    def __init__(self, *, malformed_label: bool = False) -> None:
        self.requests: list[tuple[str, Mapping[str, Any] | None]] = []
        self.malformed_label = malformed_label
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
            if "prompt" in payload:
                assert payload["add_special_tokens"] is False
                if payload["prompt"] == "post":
                    return {"tokens": [10, 12] if self.malformed_label else [10]}
                return {"tokens": [11]}
            messages = payload["messages"]
            final = messages[-1]
            content = final["content"] if final["role"] == "assistant" else None
            if content is not None and content.startswith("<action>"):
                return {"tokens": [1, 2, 3]}
            raise AssertionError(f"unexpected tokenize content {content!r}")
        if path == "/detokenize":
            assert payload is not None
            return {"prompt": "post" if payload["tokens"] == [10] else "comment"}
        if path == "/v1/chat/completions":
            assert payload is not None
            if payload.get("logprobs"):
                return {
                    "prompt_token_ids": [1, 2, 3],
                    "usage": {"prompt_tokens": 3},
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
                    ],
                }
            return {
                "prompt_token_ids": [1, 2, 3],
                "usage": {"prompt_tokens": 3},
                "choices": [
                    {
                        "message": {"content": "continued"},
                        "finish_reason": "stop",
                    }
                ],
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
    assert result.decision.probabilities == pytest.approx({"post": 0.01, "comment": 0.99})


def test_server_sample_and_alternative_order_cannot_change_client_decision() -> None:
    class ReorderedBackend(FakeVllmBackend):
        def _request(self, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
            response = super()._request(path, payload)
            if path.endswith("completions") and payload is not None and payload.get("logprobs"):
                first = response["choices"][0]["logprobs"]["content"][0]
                first["token"], first["logprob"] = "token_id:11", -3.0
                first["top_logprobs"].reverse()
            return response

    results = [
        backend.decide(
            request(),
            prior={"post": 1, "comment": 1},
            target={"post": 0.2, "comment": 0.8},
            beta=0,
            seed=1,
        )
        for backend in (FakeVllmBackend(), ReorderedBackend())
    ]
    assert results[0].score.selected_action != results[1].score.selected_action
    assert results[0].decision == results[1].decision
    for result in results:
        assert (
            tuple(result.score.raw_scores)
            == tuple(result.decision.probabilities)
            == ("post", "comment")
        )


def test_api_decision_uses_continuing_client_rng_and_validates_before_requests() -> None:
    backend = FakeVllmBackend()
    rng, expected = random.Random(7), random.Random(7)
    for _ in range(3):
        result = backend.decide(
            request(),
            prior={"post": 1, "comment": 1},
            target={"post": 0.2, "comment": 0.8},
            beta=0,
            rng=rng,
        )
        assert result.decision.random_draw == expected.random()
    assert rng.getstate() == expected.getstate()
    calls_before = len(backend.requests)
    with pytest.raises(ValueError, match="at most one"):
        backend.decide(
            request(),
            prior={"post": 1, "comment": 1},
            target={"post": 1, "comment": 1},
            beta=0,
            seed=1,
            rng=rng,
        )
    assert len(backend.requests) == calls_before


def test_vllm_api_rejects_multitoken_bare_label() -> None:
    backend = FakeVllmBackend(malformed_label=True)
    with pytest.raises(VllmApiError, match="one nonnegative token"):
        backend.score(request())


def test_vllm_api_does_not_retokenize_prefix_with_candidate() -> None:
    backend = FakeVllmBackend()
    backend.score(request())
    chat_tokenizations = [
        payload
        for path, payload in backend.requests
        if path == "/tokenize" and payload is not None and "messages" in payload
    ]
    assert len(chat_tokenizations) == 1
    assert chat_tokenizations[0]["messages"][-1]["content"] == "<action>"


@pytest.mark.parametrize("mismatch", ["ids", "count", "missing"])
def test_vllm_api_requires_preflight_generation_parity(mismatch: str) -> None:
    class MismatchedBackend(FakeVllmBackend):
        def _request(self, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
            result = super()._request(path, payload)
            if path.endswith("completions"):
                if mismatch == "ids":
                    result["prompt_token_ids"] = [1, 2, 99]
                elif mismatch == "count":
                    result["usage"] = {"prompt_tokens": 9}
                else:
                    result.pop("prompt_token_ids")
            return result

    with pytest.raises(VllmApiError, match="preflight"):
        MismatchedBackend().score(request())


@pytest.mark.parametrize("bad_score", [math.nan, math.inf, "-1.0", True, None])
def test_vllm_api_rejects_invalid_candidate_scores(bad_score: object) -> None:
    backend = FakeVllmBackend()
    response = backend._request("/v1/chat/completions", {"logprobs": True})
    response["choices"][0]["logprobs"]["content"][0]["logprob"] = bad_score
    with pytest.raises(VllmApiError, match=r"numeric|nonnumeric"):
        backend._parse_score_response(
            response,
            labels_by_action={"post": "post", "comment": "comment"},
            token_ids={"post": 10, "comment": 11},
        )


def test_vllm_api_rejects_conflicting_duplicate_candidate_score() -> None:
    backend = FakeVllmBackend()
    response = backend._request("/v1/chat/completions", {"logprobs": True})
    response["choices"][0]["logprobs"]["content"][0]["top_logprobs"][0]["logprob"] = -5
    with pytest.raises(VllmApiError, match="conflicting duplicate"):
        backend._parse_score_response(
            response,
            labels_by_action={"post": "post", "comment": "comment"},
            token_ids={"post": 10, "comment": 11},
        )


def test_vllm_api_wraps_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class InvalidResponse:
        def __enter__(self) -> InvalidResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"not-json"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: InvalidResponse())
    backend = VllmApiBackend("http://unused", model="model", logprobs_mode="raw_logprobs")
    with pytest.raises(VllmApiError, match="malformed JSON"):
        backend._request("/v1/models")


def test_vllm_api_requires_declared_raw_score_mode() -> None:
    with pytest.raises(ValueError, match="explicit"):
        VllmApiBackend("http://unused", model="model", logprobs_mode="processed_logprobs")


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
