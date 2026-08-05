"""Prior estimation and optional exact marginal calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .correction import expected_action_rates, normalize_probabilities, softmax


@dataclass(frozen=True)
class MarginalFit:
    """Action intercepts fitted to a calibration-context target."""

    intercepts: Mapping[str, float]
    achieved_rates: Mapping[str, float]
    iterations: int
    max_absolute_error: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "intercepts", MappingProxyType(dict(self.intercepts)))
        object.__setattr__(self, "achieved_rates", MappingProxyType(dict(self.achieved_rates)))


def probability_mean_prior(
    distributions: Sequence[Mapping[str, float]],
    *,
    pseudocount: float = 0.0,
) -> dict[str, float]:
    """Average categorical probabilities over representative contexts.

    A pseudocount is added per action before normalization. Unlike one sampled label per
    context, this estimator uses all scores exposed by a backend and avoids Monte Carlo noise.
    """
    if not distributions:
        raise ValueError("at least one prior context is required")
    if not math.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError("pseudocount must be finite and nonnegative")
    actions = tuple(distributions[0])
    totals = {action: pseudocount for action in actions}
    for distribution in distributions:
        if set(distribution) != set(actions):
            raise ValueError("all prior distributions must use the same actions")
        normalized = normalize_probabilities(distribution)
        for action in actions:
            totals[action] += normalized[action]
    return normalize_probabilities(totals)


def counted_prior(
    actions: Sequence[str],
    action_universe: Sequence[str],
    *,
    pseudocount: float = 1.0,
) -> dict[str, float]:
    """Paper-compatible sampled-count prior with configurable additive smoothing."""
    universe = tuple(action_universe)
    if len(universe) < 2 or len(set(universe)) != len(universe):
        raise ValueError("action_universe must contain at least two unique actions")
    if not math.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be finite and positive")
    counts = {action: pseudocount for action in universe}
    for action in actions:
        if action not in counts:
            raise ValueError(f"sampled action {action!r} is outside the action universe")
        counts[action] += 1.0
    return normalize_probabilities(counts)


def apply_intercepts(
    raw_scores: Mapping[str, float],
    intercepts: Mapping[str, float],
    beta: float,
) -> dict[str, float]:
    """Return ``softmax(beta * raw_scores + intercepts)``."""
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be finite and in [0, 1]")
    if set(raw_scores) != set(intercepts):
        raise ValueError("raw score and intercept actions must match")
    return softmax(
        {action: beta * raw_scores[action] + intercepts[action] for action in raw_scores}
    )


def fit_marginal_intercepts(
    context_scores: Sequence[Mapping[str, float]],
    target: Mapping[str, float],
    beta: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
) -> MarginalFit:
    """Fit action intercepts whose mean probabilities match ``target``.

    This iterative proportional fitting variant calibrates the finite set of supplied
    contexts. It preserves every pairwise *between-context* log-odds difference, scaled by
    ``beta``, because the fitted intercept is constant across contexts.

    This is an optional stronger calibration mode, distinct from the paper's closed form.
    Generalization to a different context distribution is not guaranteed.
    """
    if not context_scores:
        raise ValueError("at least one calibration context is required")
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be finite and in [0, 1]")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    normalized_target = normalize_probabilities(target)
    actions = tuple(normalized_target)
    for scores in context_scores:
        if set(scores) != set(actions):
            raise ValueError("all context scores must use the target actions")
        if any(not math.isfinite(value) for value in scores.values()):
            raise ValueError("context scores must be finite")

    intercepts = {action: math.log(normalized_target[action]) for action in actions}
    achieved: dict[str, float] = {}
    error = math.inf
    for iteration in range(max_iterations + 1):
        distributions = [apply_intercepts(scores, intercepts, beta) for scores in context_scores]
        achieved = expected_action_rates(distributions)
        error = max(abs(achieved[action] - normalized_target[action]) for action in actions)
        if error <= tolerance:
            return MarginalFit(intercepts, achieved, iteration, error)
        if iteration == max_iterations:
            break
        for action in actions:
            intercepts[action] += math.log(normalized_target[action]) - math.log(achieved[action])
        reference = intercepts[actions[-1]]
        intercepts = {action: value - reference for action, value in intercepts.items()}

    raise RuntimeError(
        f"marginal intercept fitting did not converge in {max_iterations} iterations; "
        f"max error={error:.6g}"
    )
