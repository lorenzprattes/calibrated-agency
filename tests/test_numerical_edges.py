"""Regression cases that used to fail despite finite, valid model inputs."""

import math
import random

import pytest

from calibrated_agency import (
    apply_intercepts,
    bayesian_correct,
    decide,
    expected_action_rates,
    fit_marginal_intercepts,
    log_softmax,
    probability_mean_prior,
    sample_probabilities,
    softmax,
)
from calibrated_agency.correction import normalize_probabilities


def test_softmax_underflow_remains_sampleable_and_averageable() -> None:
    distribution = softmax({"a": 0.0, "b": -1000.0})
    assert distribution == {"a": 1.0, "b": 0.0}
    assert sample_probabilities(distribution, rng=random.Random(1))[0] == "a"
    assert expected_action_rates([distribution, {"a": 0.0, "b": 1.0}]) == {"a": 0.5, "b": 0.5}
    assert decide({"a": 0, "b": -1000}, {"a": 1, "b": 1}, {"a": 1, "b": 1}, 1, seed=1).action == "a"


def test_sampler_rejects_all_zero_mass() -> None:
    with pytest.raises(ValueError, match="positive total"):
        sample_probabilities({"a": 0, "b": 0}, rng=random.Random(1))


def test_large_finite_weights_do_not_overflow_normalization() -> None:
    assert normalize_probabilities({"a": 1e308, "b": 1e308}) == {"a": 0.5, "b": 0.5}
    result = bayesian_correct(
        {"a": 0, "b": 0}, {"a": 1e308, "b": 1e308}, {"a": 1e308, "b": 1e308}, 0
    )
    assert result.probabilities == pytest.approx({"a": 0.5, "b": 0.5})


def test_tiny_positive_prior_is_preserved_in_log_space() -> None:
    result = bayesian_correct(
        {"a": 0, "b": 0},
        {"a": 5e-324, "b": 1e308},
        {"a": 1, "b": 1},
        1,
    )
    assert all(math.isfinite(value) for value in result.log_scores.values())
    assert result.probabilities["a"] == 1.0
    logs = log_softmax({"a": 0, "b": -1000})
    assert logs["b"] == -1000


def test_zero_prior_probabilities_can_be_smoothed() -> None:
    assert probability_mean_prior([{"a": 1, "b": 0}], pseudocount=1) == pytest.approx(
        {"a": 2 / 3, "b": 1 / 3}
    )
    with pytest.raises(ValueError, match="positive pseudocount"):
        probability_mean_prior([{"a": 1, "b": 0}])


def test_fit_handles_underflowed_initial_probabilities() -> None:
    fit = fit_marginal_intercepts([{"a": 0, "b": -1000}], {"a": 0.2, "b": 0.8}, 1)
    assert fit.achieved_rates == pytest.approx({"a": 0.2, "b": 0.8}, abs=1e-10)


def test_fit_nonconvergence_is_explicit() -> None:
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_marginal_intercepts(
            [{"a": 10, "b": 0}, {"a": -10, "b": 0}],
            {"a": 0.2, "b": 0.8},
            1,
            max_iterations=1,
        )


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_beta_zero_does_not_conceal_invalid_scores(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        apply_intercepts({"a": invalid, "b": 0}, {"a": 0, "b": 0}, 0)


def test_unrepresentable_score_range_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="floating-point capacity"):
        softmax({"a": 1e308, "b": -1e308})


def test_positive_beta_closed_form_does_not_enforce_the_marginal_target() -> None:
    contexts = [
        {"post": math.log(0.99), "comment": math.log(0.01)},
        {"post": math.log(0.2), "comment": math.log(0.8)},
    ]
    prior = probability_mean_prior([softmax(context) for context in contexts])
    target = {"post": 0.2, "comment": 0.8}
    marginal = expected_action_rates(
        [bayesian_correct(context, prior, target, 1).probabilities for context in contexts]
    )
    assert abs(marginal["post"] - target["post"]) > 0.1


def test_fitted_target_does_not_survive_context_distribution_shift() -> None:
    fit = fit_marginal_intercepts(
        [{"post": -1, "comment": 0}, {"post": 1, "comment": 0}],
        {"post": 0.2, "comment": 0.8},
        1,
    )
    assert fit.achieved_rates["post"] == pytest.approx(0.2, abs=1e-10)
    shifted = apply_intercepts({"post": 100, "comment": 0}, fit.intercepts, 1)
    assert shifted["post"] > 0.99
