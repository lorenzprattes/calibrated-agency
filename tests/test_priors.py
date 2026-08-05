from __future__ import annotations

import math

import pytest

from calibrated_agency.correction import expected_action_rates, pairwise_log_odds
from calibrated_agency.priors import (
    apply_intercepts,
    counted_prior,
    fit_marginal_intercepts,
    probability_mean_prior,
)


def test_probability_mean_prior_uses_every_distribution() -> None:
    prior = probability_mean_prior(
        [
            {"post": 0.9, "comment": 0.1},
            {"post": 0.3, "comment": 0.7},
        ]
    )
    assert prior == pytest.approx({"post": 0.6, "comment": 0.4})


def test_probability_mean_prior_normalizes_inputs_and_pseudocount() -> None:
    prior = probability_mean_prior(
        [{"a": 9, "b": 1}, {"a": 1, "b": 1}], pseudocount=1
    )
    assert prior == pytest.approx({"a": 0.6, "b": 0.4})


def test_counted_prior_matches_laplace_smoothing() -> None:
    prior = counted_prior(["post", "post", "comment"], ["post", "comment"])
    assert prior == pytest.approx({"post": 3 / 5, "comment": 2 / 5})


def test_exact_intercepts_reach_target_on_calibration_contexts() -> None:
    scores = [
        {"post": 4.0, "comment": 0.0},
        {"post": 1.0, "comment": 0.0},
        {"post": -2.0, "comment": 0.0},
        {"post": -4.0, "comment": 0.0},
    ]
    target = {"post": 0.2, "comment": 0.8}
    fit = fit_marginal_intercepts(scores, target, beta=0.6)
    achieved = expected_action_rates(
        [apply_intercepts(context, fit.intercepts, 0.6) for context in scores]
    )
    assert fit.max_absolute_error <= 1e-10
    assert achieved == pytest.approx(target, abs=1e-10)


def test_fitted_intercepts_preserve_between_context_log_odds() -> None:
    scores = [
        {"a": 2.0, "b": -1.0, "c": 0.4},
        {"a": -0.5, "b": 0.5, "c": 1.2},
    ]
    beta = 0.25
    fit = fit_marginal_intercepts(scores, {"a": 0.1, "b": 0.2, "c": 0.7}, beta)
    distributions = [apply_intercepts(context, fit.intercepts, beta) for context in scores]
    corrected_difference = pairwise_log_odds(distributions[0], "a", "c") - pairwise_log_odds(
        distributions[1], "a", "c"
    )
    raw_difference = (scores[0]["a"] - scores[0]["c"]) - (
        scores[1]["a"] - scores[1]["c"]
    )
    assert corrected_difference == pytest.approx(beta * raw_difference)


def test_exact_intercepts_handle_multiclass_extreme_scores() -> None:
    scores = [
        {"a": 20.0, "b": 0.0, "c": -20.0},
        {"a": -20.0, "b": 20.0, "c": 0.0},
        {"a": 0.0, "b": -20.0, "c": 20.0},
        {"a": 4.0, "b": 2.0, "c": -3.0},
    ]
    target = {"a": 0.01, "b": 0.19, "c": 0.80}
    fit = fit_marginal_intercepts(
        scores, target, beta=0.6, tolerance=1e-9, max_iterations=50_000
    )
    assert fit.achieved_rates == pytest.approx(target, abs=1e-9)


def test_beta_zero_fit_is_immediate() -> None:
    fit = fit_marginal_intercepts(
        [{"a": 100.0, "b": -100.0}], {"a": 0.3, "b": 0.7}, beta=0
    )
    assert fit.iterations == 0
    assert fit.achieved_rates == pytest.approx({"a": 0.3, "b": 0.7})


@pytest.mark.parametrize("pseudocount", [0, -1, math.nan])
def test_counted_prior_requires_positive_smoothing(pseudocount: float) -> None:
    with pytest.raises(ValueError, match="pseudocount"):
        counted_prior([], ["a", "b"], pseudocount=pseudocount)


def test_prior_estimators_reject_empty_or_mismatched_data() -> None:
    with pytest.raises(ValueError, match="at least one"):
        probability_mean_prior([])
    with pytest.raises(ValueError, match="pseudocount"):
        probability_mean_prior([{"a": 1, "b": 1}], pseudocount=-1)
    with pytest.raises(ValueError, match="same actions"):
        probability_mean_prior([{"a": 1, "b": 1}, {"a": 1, "c": 1}])
    with pytest.raises(ValueError, match="outside"):
        counted_prior(["c"], ["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        counted_prior([], ["a", "a"])


def test_apply_and_fit_validation_errors() -> None:
    with pytest.raises(ValueError, match="beta"):
        apply_intercepts({"a": 0, "b": 1}, {"a": 0, "b": 0}, 2)
    with pytest.raises(ValueError, match="match"):
        apply_intercepts({"a": 0, "b": 1}, {"a": 0, "c": 0}, 0.5)
    with pytest.raises(ValueError, match="at least one"):
        fit_marginal_intercepts([], {"a": 1, "b": 1}, 0.5)
    with pytest.raises(ValueError, match="tolerance"):
        fit_marginal_intercepts([{"a": 0, "b": 0}], {"a": 1, "b": 1}, 0.5, tolerance=0)
    with pytest.raises(ValueError, match="positive"):
        fit_marginal_intercepts(
            [{"a": 0, "b": 0}], {"a": 1, "b": 1}, 0.5, max_iterations=0
        )
    with pytest.raises(ValueError, match="target actions"):
        fit_marginal_intercepts([{"a": 0, "c": 0}], {"a": 1, "b": 1}, 0.5)
