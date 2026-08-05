from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from calibrated_agency.fingerprint import (
    canonical_json,
    file_sha256,
    fingerprint,
    prior_cache_identity,
)
from calibrated_agency.schemas import (
    ActionSpace,
    Candidate,
    ChatMessage,
    ScoreRequest,
    ScoreResult,
)


def test_action_space_preserves_order_and_maps_labels() -> None:
    space = ActionSpace.from_mapping({"post": "P", "comment": "C"})
    assert space.actions == ("post", "comment")
    assert space.labels == ("P", "C")
    assert space.action_for_label("C") == "comment"
    assert space.label_for_action("post") == "P"


def test_action_space_rejects_invalid_candidates_and_lookups() -> None:
    with pytest.raises(ValueError, match="action"):
        Candidate("", "1")
    with pytest.raises(ValueError, match="label"):
        Candidate("a", "")
    with pytest.raises(ValueError, match="at least two"):
        ActionSpace((Candidate("a", "1"),))
    with pytest.raises(ValueError, match="actions"):
        ActionSpace((Candidate("a", "1"), Candidate("a", "2")))
    with pytest.raises(ValueError, match="labels"):
        ActionSpace((Candidate("a", "1"), Candidate("b", "1")))
    space = ActionSpace.from_mapping({"a": "1", "b": "2"})
    with pytest.raises(KeyError, match="unknown"):
        space.action_for_label("3")
    with pytest.raises(KeyError, match="unknown"):
        space.label_for_action("c")


def test_score_result_is_validated_and_immutable() -> None:
    result = ScoreResult(
        raw_scores={"a": 0.0, "b": -1.0},
        probabilities={"a": 0.7, "b": 0.3},
        token_ids={"a": 1, "b": 2},
    )
    with pytest.raises(TypeError):
        result.raw_scores["a"] = 4.0  # type: ignore[index]
    with pytest.raises(ValueError, match="sum"):
        ScoreResult(
            raw_scores={"a": 0.0, "b": -1.0},
            probabilities={"a": 0.5, "b": 0.3},
            token_ids={"a": 1, "b": 2},
        )
    with pytest.raises(ValueError, match="finite"):
        ScoreResult(
            raw_scores={"a": math.nan, "b": -1.0},
            probabilities={"a": 0.5, "b": 0.5},
            token_ids={"a": 1, "b": 2},
        )


def test_request_requires_messages_and_nonempty_prefix() -> None:
    with pytest.raises(ValueError, match="role"):
        ChatMessage("", "hello")
    space = ActionSpace.from_mapping({"a": "1", "b": "2"})
    with pytest.raises(ValueError, match="message"):
        ScoreRequest((), space)
    with pytest.raises(ValueError, match="prefix"):
        ScoreRequest((ChatMessage("user", "hello"),), space, assistant_prefix="")


def test_fingerprint_is_canonical() -> None:
    first = {"b": [2, 1], "a": {"z": True}}
    second = {"a": {"z": True}, "b": [2, 1]}
    assert canonical_json(first) == canonical_json(second)
    assert fingerprint(first) == fingerprint(second)
    assert fingerprint({"x": 1}) != fingerprint({"x": 2})


def test_fingerprint_supports_dataclasses_paths_and_sets(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Value:
        path: Path
        labels: set[str]

    first = Value(Path("fixture.json"), {"b", "a"})
    assert canonical_json(first) == '{"labels":["a","b"],"path":"fixture.json"}'
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("content", encoding="utf-8")
    assert file_sha256(fixture) == (
        "ed7002b439e9ac845f22357d822bac1444730fbdb6016d3ec9432297b9ec9f73"
    )
    with pytest.raises(TypeError, match="unsupported"):
        canonical_json(object())
    with pytest.raises(TypeError, match="keys"):
        canonical_json({1: "integer key"})
    with pytest.raises(ValueError, match="finite"):
        canonical_json({"value": math.inf})


def test_prior_cache_identity_changes_on_menu_or_temperature() -> None:
    base = dict(
        backend="vllm_api",
        backend_version="0.26.0",
        model="model",
        model_revision="abc",
        tokenizer_revision="abc",
        chat_template_hash="template",
        prompt_hash="prompt",
        action_schema=[{"action": "a", "label": "1"}, {"action": "b", "label": "2"}],
        token_ids={"a": 1, "b": 2},
        context_dataset_hash="contexts",
        estimator="probability_mean",
        estimator_settings={"pseudocount": 0.0, "context_seed": 12345},
        sample_count=10,
        score_mode="raw_logprobs",
        generation_settings={"temperature": 1.0},
    )
    first = prior_cache_identity(**base)
    changed = prior_cache_identity(
        **{**base, "generation_settings": {"temperature": 0.7}}
    )
    assert fingerprint(first) != fingerprint(changed)
    changed_estimator = prior_cache_identity(
        **{
            **base,
            "estimator_settings": {"pseudocount": 1.0, "context_seed": 12345},
        }
    )
    assert fingerprint(first) != fingerprint(changed_estimator)
    with pytest.raises(ValueError, match="positive"):
        prior_cache_identity(**{**base, "sample_count": 0})
