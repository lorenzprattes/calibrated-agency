from __future__ import annotations

import math
import random

import pytest

from calibrated_agency.correction import (
    bayesian_correct,
    contextual_log_odds,
    decide,
    expected_action_rates,
    normalize_probabilities,
    pairwise_log_odds,
    sample_probabilities,
    softmax,
)


def test_beta_zero_is_exact_target() -> None:
    result = bayesian_correct(
        {"post": 12.0, "comment": -7.0},
        {"post": 0.8, "comment": 0.2},
        {"post": 0.13, "comment": 0.87},
        beta=0.0,
    )
    assert result.probabilities == pytest.approx({"post": 0.13, "comment": 0.87})


def test_beta_one_matches_formula() -> None:
    result = bayesian_correct(
        {"post": 2.0, "comment": 0.5},
        {"post": 0.6, "comment": 0.4},
        {"post": 0.2, "comment": 0.8},
        beta=1.0,
    )
    expected_log_odds = 1.5 - math.log(0.6 / 0.4) + math.log(0.2 / 0.8)
    observed_log_odds = pairwise_log_odds(result.probabilities, "post", "comment")
    assert observed_log_odds == pytest.approx(expected_log_odds)


def test_raw_score_additive_constant_is_irrelevant() -> None:
    arguments = ({"a": 0.7, "b": -1.2, "c": 0.3}, {"a": 2, "b": 3, "c": 5})
    first = bayesian_correct(arguments[0], arguments[1], {"a": 1, "b": 1, "c": 2}, 0.4)
    shifted = bayesian_correct(
        {action: score + 1000 for action, score in arguments[0].items()},
        arguments[1],
        {"a": 1, "b": 1, "c": 2},
        0.4,
    )
    assert first.probabilities == pytest.approx(shifted.probabilities)


def test_between_context_log_odds_are_scaled_by_beta() -> None:
    prior = {"post": 0.55, "comment": 0.45}
    target = {"post": 0.1, "comment": 0.9}
    first_raw = {"post": 2.0, "comment": 0.0}
    second_raw = {"post": -0.5, "comment": 0.25}
    beta = 0.15
    first = bayesian_correct(first_raw, prior, target, beta)
    second = bayesian_correct(second_raw, prior, target, beta)
    corrected_difference = pairwise_log_odds(
        first.probabilities, "post", "comment"
    ) - pairwise_log_odds(second.probabilities, "post", "comment")
    raw_difference = (first_raw["post"] - first_raw["comment"]) - (
        second_raw["post"] - second_raw["comment"]
    )
    assert corrected_difference == pytest.approx(beta * raw_difference)


def test_contextual_log_odds_centers_on_prior() -> None:
    value = contextual_log_odds(
        {"post": math.log(0.8), "comment": math.log(0.2)},
        {"post": 0.6, "comment": 0.4},
        "post",
        "comment",
    )
    assert value == pytest.approx(math.log(4) - math.log(1.5))


def test_seeded_decision_and_sampler_are_deterministic() -> None:
    decision = decide(
        {"left": 0.0, "right": 0.0},
        {"left": 0.5, "right": 0.5},
        {"left": 0.25, "right": 0.75},
        0.0,
        seed=7,
    )
    action, draw = sample_probabilities(
        {"left": 0.25, "right": 0.75}, rng=random.Random(7)
    )
    assert (decision.action, decision.random_draw) == (action, draw)


@pytest.mark.parametrize("beta", [-0.1, 1.1, math.nan])
def test_invalid_beta_is_rejected(beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        bayesian_correct({"a": 0.0, "b": 1.0}, {"a": 1, "b": 1}, {"a": 1, "b": 1}, beta)


def test_action_mismatch_and_zero_weight_are_rejected() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        bayesian_correct({"a": 0.0, "b": 1.0}, {"a": 1, "b": 1}, {"a": 1, "c": 1}, 1)
    with pytest.raises(ValueError, match="strictly positive"):
        bayesian_correct({"a": 0.0, "b": 1.0}, {"a": 1, "b": 0}, {"a": 1, "b": 1}, 1)


def test_softmax_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        softmax({"a": 0.0, "b": math.inf})


def test_probability_normalization_rejects_invalid_shapes_and_values() -> None:
    with pytest.raises(ValueError, match="at least two"):
        normalize_probabilities({"a": 1})
    with pytest.raises(ValueError, match="strictly positive"):
        normalize_probabilities({"a": -1, "b": 2})


def test_expected_rates_validate_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        expected_action_rates([])
    with pytest.raises(ValueError, match="at least two"):
        expected_action_rates([{"a": 1.0}])
    with pytest.raises(ValueError, match="same actions"):
        expected_action_rates([{"a": 1, "b": 1}, {"a": 1, "c": 1}])
