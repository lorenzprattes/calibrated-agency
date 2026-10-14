"""The next generated token does not retokenize an existing prefix."""

from typing import Any

import pytest

from calibrated_agency.backends.tokenization import resolve_action_tokens
from calibrated_agency.backends.transformers_native import TransformersNativeBackend
from calibrated_agency.backends.vllm_native import VllmNativeBackend
from calibrated_agency.schemas import ActionSpace, ChatMessage, ScoreRequest


class MergingTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return {"post": [10], "comment": [11]}[text]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return {10: "post", 11: "comment"}[ids[0]]

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
        # Retokenizing the combined string changes a previous prefix token.
        if messages[-1]["content"] == "<action>post":
            return [1, 99]
        if messages[-1]["content"] == "<action>comment":
            return [1, 98]
        return [1, 2]


class FakeModel:
    def __call__(self) -> None:
        pass

    def get_input_embeddings(self) -> None:
        pass


class FakeLlm:
    def generate(self) -> None:
        pass

    def get_tokenizer(self) -> MergingTokenizer:
        return MergingTokenizer()


def test_native_backends_share_bare_label_identity_even_when_prefix_merges() -> None:
    request = ScoreRequest(
        (ChatMessage("user", "Choose"),),
        ActionSpace.from_mapping({"post": "post", "comment": "comment"}),
        assistant_prefix="<action>",
    )
    transformers = TransformersNativeBackend(FakeModel(), MergingTokenizer())
    vllm = VllmNativeBackend(FakeLlm(), model_id="test", logprobs_mode="raw_logprobs")
    for backend in (transformers, vllm):
        prefix, candidates = backend.resolve_token_ids(request)
        assert prefix == [1, 2]
        assert candidates == {"post": 10, "comment": 11}


@pytest.mark.parametrize("ids", [[1, 2], [], [-1], [True]])
def test_candidate_identity_rejects_invalid_tokenization(ids: list[int]) -> None:
    space = ActionSpace.from_mapping({"a": "A", "b": "B"})
    with pytest.raises(RuntimeError, match="one nonnegative"):
        resolve_action_tokens(space, lambda text: ids, lambda tokens: "A")


def test_candidate_identity_rejects_wrong_decoding() -> None:
    space = ActionSpace.from_mapping({"a": "A", "b": "B"})
    with pytest.raises(RuntimeError, match="expected exact"):
        resolve_action_tokens(space, lambda text: [1], lambda ids: " A")


def test_candidate_identity_rejects_alias_collision() -> None:
    space = ActionSpace.from_mapping({"a": "A", "b": "B"})
    count = iter(["A", "B"])
    with pytest.raises(RuntimeError, match="duplicate"):
        resolve_action_tokens(space, lambda text: [1], lambda ids: next(count))


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_native_decisions_reject_non_sampling_temperatures(temperature: float) -> None:
    request = ScoreRequest(
        (ChatMessage("user", "Choose"),),
        ActionSpace.from_mapping({"post": "post", "comment": "comment"}),
    )
    transformers = TransformersNativeBackend(FakeModel(), MergingTokenizer())
    vllm = VllmNativeBackend(FakeLlm(), model_id="test", logprobs_mode="raw_logprobs")
    for backend in (transformers, vllm):
        with pytest.raises(ValueError, match="finite and positive"):
            backend.decide_native(
                request,
                prior={"post": 1, "comment": 1},
                target={"post": 1, "comment": 1},
                beta=0,
                temperature=temperature,
            )
