"""Backend-independent Bayesian action correction."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .schemas import DecisionResult, _validate_distribution


@dataclass(frozen=True)
class CorrectionResult:
    log_scores: Mapping[str, float]
    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        _validate_distribution(self.log_scores, self.probabilities)
        object.__setattr__(self, "log_scores", MappingProxyType(dict(self.log_scores)))
        object.__setattr__(
            self,
            "probabilities",
            MappingProxyType({action: self.probabilities[action] for action in self.log_scores}),
        )


def _require_same_keys(reference: Mapping[str, object], **others: Mapping[str, object]) -> None:
    expected = set(reference)
    if len(expected) < 2:
        raise ValueError("a categorical distribution requires at least two actions")
    for name, values in others.items():
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ValueError(f"{name} action mismatch: missing={missing}, extra={extra}")


def normalize_probabilities(
    weights: Mapping[str, float], *, allow_zero: bool = False
) -> dict[str, float]:
    """Normalize finite weights without overflowing their sum.

    Priors and targets must be positive. Sampled/averaged distributions may
    contain zeros, including zeros produced by floating-point softmax underflow.
    """
    if len(weights) < 2:
        raise ValueError("a categorical distribution requires at least two actions")
    if any(
        not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero)
        for value in weights.values()
    ):
        requirement = "nonnegative" if allow_zero else "strictly positive"
        raise ValueError(f"probability weights must be finite and {requirement}")
    peak = max(weights.values())
    if peak == 0:
        raise ValueError("probability weights must have a positive total")
    scaled = {action: value / peak for action, value in weights.items()}
    total = math.fsum(scaled.values())
    return {action: value / total for action, value in scaled.items()}


def log_normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize positive weights in log space, retaining subnormal weights."""
    # Validate without taking logarithms of rounded normalized probabilities.
    normalize_probabilities(weights)
    logs = {action: math.log(value) for action, value in weights.items()}
    return log_softmax(logs)


def log_softmax(scores: Mapping[str, float]) -> dict[str, float]:
    """Stable log probabilities without exponentiating the final result."""
    if len(scores) < 2:
        raise ValueError("softmax requires at least two actions")
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("scores must be finite")
    maximum = max(scores.values())
    centered = {action: value - maximum for action, value in scores.items()}
    if any(not math.isfinite(value) for value in centered.values()):
        raise ValueError("score range exceeds floating-point capacity")
    log_total = math.log(math.fsum(math.exp(value) for value in centered.values()))
    return {action: value - log_total for action, value in centered.items()}


def softmax(scores: Mapping[str, float]) -> dict[str, float]:
    """Stable softmax over a finite action map."""
    return {action: math.exp(value) for action, value in log_softmax(scores).items()}


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
    return bayesian_correct_log(
        raw_scores, log_normalize_weights(prior), log_normalize_weights(target), beta
    )


def bayesian_correct_log(
    raw_scores: Mapping[str, float],
    log_prior: Mapping[str, float],
    log_target: Mapping[str, float],
    beta: float,
) -> CorrectionResult:
    """Correct from finite log weights without exponentiating priors or targets.

    Log weights need not be normalized. Output order follows ``raw_scores``.
    Negative infinity is rejected; callers own any zero-weight flooring policy.
    """
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be finite and in [0, 1]")
    _require_same_keys(raw_scores, log_prior=log_prior, log_target=log_target)
    if any(not math.isfinite(value) for value in raw_scores.values()):
        raise ValueError("raw scores must be finite")
    log_prior = log_softmax(log_prior)
    log_target = log_softmax(log_target)
    corrected = {
        action: beta * (raw_scores[action] - log_prior[action]) + log_target[action]
        for action in raw_scores
    }
    return CorrectionResult(corrected, softmax(corrected))


def sample_probabilities(
    probabilities: Mapping[str, float],
    *,
    rng: random.Random,
) -> tuple[str, float]:
    """Sample in mapping order and return both the action and uniform draw."""
    normalized = normalize_probabilities(probabilities, allow_zero=True)
    draw = rng.random()
    cumulative = 0.0
    last_action = next(action for action in reversed(normalized) if normalized[action] > 0)
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
    seed: int | None = None,
    rng: random.Random | None = None,
) -> DecisionResult:
    """Correct and sample using exactly one of ``seed`` or a continuing ``rng``."""
    if (seed is None) == (rng is None):
        raise ValueError("provide exactly one of seed or rng")
    correction = bayesian_correct(raw_scores, prior, target, beta)
    action, draw = sample_probabilities(
        correction.probabilities, rng=rng if rng is not None else random.Random(seed)
    )
    return DecisionResult(action, correction.probabilities, correction.log_scores, draw)


def pairwise_log_odds(distribution: Mapping[str, float], first: str, second: str) -> float:
    log_probabilities = log_normalize_weights(distribution)
    return log_probabilities[first] - log_probabilities[second]


def contextual_log_odds(
    raw_scores: Mapping[str, float],
    prior: Mapping[str, float],
    first: str,
    second: str,
) -> float:
    """Pairwise model log odds centered on its marginal prior."""
    _require_same_keys(raw_scores, prior=prior)
    if any(not math.isfinite(value) for value in raw_scores.values()):
        raise ValueError("raw scores must be finite")
    log_prior = log_normalize_weights(prior)
    return (raw_scores[first] - raw_scores[second]) - (log_prior[first] - log_prior[second])


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
        normalized = normalize_probabilities(distribution, allow_zero=True)
        for action, probability in normalized.items():
            totals[action] += probability
    return {action: total / len(distributions) for action, total in totals.items()}
