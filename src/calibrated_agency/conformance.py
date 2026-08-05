"""Backend conformance and parity checks."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .backends.base import ActionScorer
from .correction import bayesian_correct
from .schemas import ScoreRequest, ScoreResult


@dataclass(frozen=True)
class ParityReport:
    contexts: int
    pairwise_comparisons: int
    max_absolute_log_odds_difference: float
    mean_absolute_log_odds_difference: float
    max_corrected_probability_difference: float
    token_id_mismatches: int
    passed: bool


def _pairwise_differences(result: ScoreResult) -> dict[tuple[str, str], float]:
    actions = tuple(result.raw_scores)
    return {
        (first, second): result.raw_scores[first] - result.raw_scores[second]
        for first, second in itertools.combinations(actions, 2)
    }


def compare_backends(
    first: ActionScorer,
    second: ActionScorer,
    requests: Sequence[ScoreRequest],
    *,
    prior: Mapping[str, float],
    target: Mapping[str, float],
    betas: Sequence[float] = (0.0, 0.15, 0.5, 1.0),
    log_odds_tolerance: float = 1e-3,
    probability_tolerance: float = 1e-4,
) -> ParityReport:
    """Compare backend-invariant score differences and corrected probabilities."""
    if not requests:
        raise ValueError("at least one score request is required")
    if log_odds_tolerance < 0 or probability_tolerance < 0:
        raise ValueError("tolerances must be nonnegative")
    absolute_log_odds_differences: list[float] = []
    max_probability_difference = 0.0
    token_id_mismatches = 0

    for request in requests:
        first_result = first.score(request)
        second_result = second.score(request)
        first_odds = _pairwise_differences(first_result)
        second_odds = _pairwise_differences(second_result)
        if set(first_odds) != set(second_odds):
            raise ValueError("backends returned different action sets")
        absolute_log_odds_differences.extend(
            abs(first_odds[pair] - second_odds[pair]) for pair in first_odds
        )
        if dict(first_result.token_ids) != dict(second_result.token_ids):
            token_id_mismatches += 1
        for beta in betas:
            first_corrected = bayesian_correct(first_result.raw_scores, prior, target, beta)
            second_corrected = bayesian_correct(second_result.raw_scores, prior, target, beta)
            for action in first_corrected.probabilities:
                difference = abs(
                    first_corrected.probabilities[action]
                    - second_corrected.probabilities[action]
                )
                max_probability_difference = max(max_probability_difference, difference)

    max_log_odds = max(absolute_log_odds_differences)
    mean_log_odds = math.fsum(absolute_log_odds_differences) / len(
        absolute_log_odds_differences
    )
    passed = (
        max_log_odds <= log_odds_tolerance
        and max_probability_difference <= probability_tolerance
        and token_id_mismatches == 0
    )
    return ParityReport(
        contexts=len(requests),
        pairwise_comparisons=len(absolute_log_odds_differences),
        max_absolute_log_odds_difference=max_log_odds,
        mean_absolute_log_odds_difference=mean_log_odds,
        max_corrected_probability_difference=max_probability_difference,
        token_id_mismatches=token_id_mismatches,
        passed=passed,
    )
