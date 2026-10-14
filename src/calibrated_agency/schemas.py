"""Validated, backend-independent request and result types."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _validate_distribution(scores: Mapping[str, float], probabilities: Mapping[str, float]) -> None:
    # Import on use: correction constructs these schema objects itself.
    from .correction import softmax

    if set(scores) != set(probabilities):
        raise ValueError("score and probability keys must match")
    if len(scores) < 2:
        raise ValueError("a distribution requires at least two actions")
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("scores must be finite")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not math.isclose(math.fsum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")
    expected = softmax(scores)
    if any(
        not math.isclose(probabilities[action], expected[action], rel_tol=1e-9, abs_tol=1e-12)
        for action in scores
    ):
        raise ValueError("probabilities must match the softmax of scores")


@dataclass(frozen=True)
class Candidate:
    """A semantic action and the exact text used to represent it to a model."""

    action: str
    label: str

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("candidate action must not be empty")
        if not self.label:
            raise ValueError("candidate label must not be empty")


@dataclass(frozen=True)
class ActionSpace:
    """An ordered finite action set.

    Order is part of the decision schema and therefore part of cache/provenance identity.
    """

    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if len(self.candidates) < 2:
            raise ValueError("an action space requires at least two candidates")
        actions = self.actions
        labels = self.labels
        if len(set(actions)) != len(actions):
            raise ValueError("candidate actions must be unique")
        if len(set(labels)) != len(labels):
            raise ValueError("candidate labels must be unique")

    @classmethod
    def from_mapping(cls, labels_by_action: Mapping[str, str]) -> ActionSpace:
        return cls(tuple(Candidate(action, label) for action, label in labels_by_action.items()))

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(candidate.action for candidate in self.candidates)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(candidate.label for candidate in self.candidates)

    def action_for_label(self, label: str) -> str:
        for candidate in self.candidates:
            if candidate.label == label:
                return candidate.action
        raise KeyError(f"unknown candidate label {label!r}")

    def label_for_action(self, action: str) -> str:
        for candidate in self.candidates:
            if candidate.action == action:
                return candidate.label
        raise KeyError(f"unknown action {action!r}")


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("message role must not be empty")

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ScoreRequest:
    """A categorical decision at the next autoregressive token."""

    messages: tuple[ChatMessage, ...]
    action_space: ActionSpace
    assistant_prefix: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages:
            raise ValueError("score request requires at least one message")
        if self.assistant_prefix == "":
            raise ValueError("assistant_prefix must be nonempty or None")


@dataclass(frozen=True)
class ScoreResult:
    """Raw candidate scores plus their normalization over the declared action set."""

    raw_scores: Mapping[str, float]
    probabilities: Mapping[str, float]
    token_ids: Mapping[str, int]
    selected_action: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_response: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        keys = set(self.raw_scores)
        if keys != set(self.probabilities) or keys != set(self.token_ids):
            raise ValueError("score, probability, and token-id keys must match")
        if len(keys) < 2:
            raise ValueError("score result requires at least two actions")
        _validate_distribution(self.raw_scores, self.probabilities)
        if len(set(self.token_ids.values())) != len(self.token_ids):
            raise ValueError("candidate token IDs must be distinct")
        if any(type(value) is not int or value < 0 for value in self.token_ids.values()):
            raise ValueError("candidate token IDs must be nonnegative integers")
        if self.selected_action is not None and self.selected_action not in keys:
            raise ValueError("selected action is outside the action set")
        if self.selected_action is not None and self.probabilities[self.selected_action] <= 0:
            raise ValueError("selected action must have positive probability")

        object.__setattr__(
            self,
            "raw_scores",
            MappingProxyType({action: self.raw_scores[action] for action in self.token_ids}),
        )
        object.__setattr__(
            self,
            "probabilities",
            MappingProxyType({action: self.probabilities[action] for action in self.token_ids}),
        )
        object.__setattr__(self, "token_ids", MappingProxyType(dict(self.token_ids)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.raw_response is not None:
            object.__setattr__(self, "raw_response", MappingProxyType(dict(self.raw_response)))


@dataclass(frozen=True)
class DecisionResult:
    action: str
    probabilities: Mapping[str, float]
    corrected_log_scores: Mapping[str, float]
    random_draw: float

    def __post_init__(self) -> None:
        _validate_distribution(self.corrected_log_scores, self.probabilities)
        if self.action not in self.probabilities:
            raise ValueError("selected action is outside the distribution")
        if self.probabilities[self.action] <= 0:
            raise ValueError("selected action must have positive probability")
        if not math.isfinite(self.random_draw) or not 0 <= self.random_draw < 1:
            raise ValueError("random_draw must be finite and in [0, 1)")
        object.__setattr__(
            self,
            "probabilities",
            MappingProxyType(
                {action: self.probabilities[action] for action in self.corrected_log_scores}
            ),
        )
        object.__setattr__(
            self, "corrected_log_scores", MappingProxyType(dict(self.corrected_log_scores))
        )


@dataclass(frozen=True)
class ScoredDecision:
    """A backend score response and the corrected client-side decision derived from it."""

    score: ScoreResult
    decision: DecisionResult

    def __post_init__(self) -> None:
        if set(self.score.raw_scores) != set(self.decision.probabilities):
            raise ValueError("score and decision action sets must match")
