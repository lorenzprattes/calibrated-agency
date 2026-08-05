"""Backend-independent Bayesian action correction."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .schemas import DecisionResult


@dataclass(frozen=True)
class CorrectionResult:
    log_scores: Mapping[str, float]
    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_scores", MappingProxyType(dict(self.log_scores)))
        object.__setattr__(self, "probabilities", MappingProxyType(dict(self.probabilities)))


def _require_same_keys(reference: Mapping[str, object], **others: Mapping[str, object]) -> None:
    expected = set(reference)
    if len(expected) < 2:
        raise ValueError("a categorical distribution requires at least two actions")
    for name, values in others.items():
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ValueError(f"{name} action mismatch: missing={missing}, extra={extra}")


def normalize_probabilities(weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize strictly positive finite weights."""
    if len(weights) < 2:
        raise ValueError("a categorical distribution requires at least two actions")
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("probability weights must be finite and strictly positive")
    total = math.fsum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("probability weights have an invalid total")
    return {action: value / total for action, value in weights.items()}


def softmax(scores: Mapping[str, float]) -> dict[str, float]:
    """Stable softmax over a finite action map."""
    if len(scores) < 2:
        raise ValueError("softmax requires at least two actions")
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("scores must be finite")
    maximum = max(scores.values())
    weights = {action: math.exp(value - maximum) for action, value in scores.items()}
    total = math.fsum(weights.values())
    return {action: value / total for action, value in weights.items()}


def bayesian_correct(
    raw_scores: Mapping[str, float],
    prior: Mapping[str, float],
    target: Mapping[str, float],
    beta: float,
) -> CorrectionResult:
    """Apply the paper's closed-form categorical correction.

    ``raw_scores`` may be raw logits or log probabilities: additive constants cancel.
    ``prior`` and ``target`` are positive weights and are normalized internally.

    The result is

    ``log q(a|x) ∝ beta * (raw(a|x) - log prior(a)) + log target(a)``.
    """
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be finite and in [0, 1]")
    _require_same_keys(raw_scores, prior=prior, target=target)
    if any(not math.isfinite(value) for value in raw_scores.values()):
        raise ValueError("raw scores must be finite")
    normalized_prior = normalize_probabilities(prior)
    normalized_target = normalize_probabilities(target)
    corrected = {
        action: beta * (raw_scores[action] - math.log(normalized_prior[action]))
        + math.log(normalized_target[action])
        for action in raw_scores
    }
    return CorrectionResult(corrected, softmax(corrected))


def sample_probabilities(
    probabilities: Mapping[str, float],
    *,
    rng: random.Random,
) -> tuple[str, float]:
    """Sample in mapping order and return both the action and uniform draw."""
    normalized = normalize_probabilities(probabilities)
    draw = rng.random()
    cumulative = 0.0
    last_action = next(reversed(normalized))
    for action, probability in normalized.items():
        cumulative += probability
        if draw < cumulative:
            return action, draw
    return last_action, draw


def decide(
    raw_scores: Mapping[str, float],
    prior: Mapping[str, float],
    target: Mapping[str, float],
    beta: float,
    *,
    seed: int,
) -> DecisionResult:
    correction = bayesian_correct(raw_scores, prior, target, beta)
    action, draw = sample_probabilities(correction.probabilities, rng=random.Random(seed))
    return DecisionResult(action, correction.probabilities, correction.log_scores, draw)


def pairwise_log_odds(distribution: Mapping[str, float], first: str, second: str) -> float:
    normalized = normalize_probabilities(distribution)
    return math.log(normalized[first]) - math.log(normalized[second])


def contextual_log_odds(
    raw_scores: Mapping[str, float],
    prior: Mapping[str, float],
    first: str,
    second: str,
) -> float:
    """Pairwise model log odds centered on its marginal prior."""
    _require_same_keys(raw_scores, prior=prior)
    normalized_prior = normalize_probabilities(prior)
    return (raw_scores[first] - raw_scores[second]) - (
        math.log(normalized_prior[first]) - math.log(normalized_prior[second])
    )


def expected_action_rates(distributions: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not distributions:
        raise ValueError("at least one distribution is required")
    actions = set(distributions[0])
    if len(actions) < 2:
        raise ValueError("distributions require at least two actions")
    for distribution in distributions:
        if set(distribution) != actions:
            raise ValueError("all distributions must use the same actions")
    totals = {action: 0.0 for action in distributions[0]}
    for distribution in distributions:
        normalized = normalize_probabilities(distribution)
        for action, probability in normalized.items():
            totals[action] += probability
    return {action: total / len(distributions) for action, total in totals.items()}
