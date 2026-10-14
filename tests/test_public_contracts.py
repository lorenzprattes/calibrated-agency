"""Focused regressions for public correction, sampling, and result contracts."""

import asyncio
import math
import random

import pytest

from calibrated_agency import (
    ActionSpace,
    AsyncActionScorer,
    ChatMessage,
    CorrectionResult,
    DecisionResult,
    ScoredDecision,
    ScoreRequest,
    ScoreResult,
    bayesian_correct,
    bayesian_correct_log,
    decide,
)


def test_log_correction_matches_weight_correction_and_preserves_input_order() -> None:
    scores = {"post": 2.0, "reply": -1.0, "stop": 0.0}
    prior = {"stop": 3.0, "reply": 2.0, "post": 1.0}
    target = {"post": 2.0, "reply": 4.0, "stop": 1.0}
    for beta in (0, 0.15, 0.5, 1):
        weighted = bayesian_correct(scores, prior, target, beta)
        logged = bayesian_correct_log(
            scores,
            {a: math.log(p) for a, p in prior.items()},
            {a: math.log(p) for a, p in target.items()},
            beta,
        )
        assert logged.probabilities == pytest.approx(weighted.probabilities)
        assert tuple(logged.probabilities) == tuple(scores)


def test_log_correction_preserves_prior_below_probability_underflow() -> None:
    result = bayesian_correct_log(
        {"a": -1000, "b": 0},
        {"a": -1000, "b": 0},
        {"a": 0, "b": 0},
        1,
    )
    assert result.probabilities == pytest.approx({"a": 0.5, "b": 0.5})


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_log_correction_rejects_nonfinite_log_weights(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        bayesian_correct_log({"a": 0, "b": 0}, {"a": invalid, "b": 0}, {"a": 0, "b": 0}, 0)


def test_decide_advances_exactly_one_caller_rng_draw_per_decision() -> None:
    rng, expected = random.Random(7), random.Random(7)
    for _ in range(5):
        choice = decide({"a": 0, "b": 0}, {"a": 1, "b": 1}, {"a": 1, "b": 3}, 0, rng=rng)
        draw = expected.random()
        assert choice.random_draw == draw
        assert choice.action == ("a" if draw < 0.25 else "b")
    assert rng.getstate() == expected.getstate()


def test_decide_rejects_missing_or_ambiguous_rng_without_consuming_it() -> None:
    rng = random.Random(7)
    before = rng.getstate()
    with pytest.raises(ValueError, match="exactly one"):
        decide({"a": 0, "b": 0}, {"a": 1, "b": 1}, {"a": 1, "b": 1}, 0, seed=7, rng=rng)
    assert rng.getstate() == before
    with pytest.raises(ValueError, match="exactly one"):
        decide({"a": 0, "b": 0}, {"a": 1, "b": 1}, {"a": 1, "b": 1}, 0)


def test_score_results_follow_token_map_order_and_reject_inconsistent_probabilities() -> None:
    result = ScoreResult(
        {"b": 0, "a": 0},
        {"b": 0.5, "a": 0.5},
        {"a": 1, "b": 2},
    )
    assert tuple(result.raw_scores) == tuple(result.probabilities) == ("a", "b")
    with pytest.raises(ValueError, match="softmax"):
        ScoreResult({"a": 0, "b": 0}, {"a": 0.2, "b": 0.8}, {"a": 1, "b": 2})
    with pytest.raises(ValueError, match="softmax"):
        CorrectionResult({"a": 0, "b": 0}, {"a": 0.2, "b": 0.8})


@pytest.mark.parametrize("draw", [-0.1, 1, math.nan, math.inf])
def test_decision_result_rejects_invalid_draw(draw: float) -> None:
    with pytest.raises(ValueError, match="random_draw"):
        DecisionResult("a", {"a": 0.5, "b": 0.5}, {"a": 0, "b": 0}, draw)


def test_decision_results_require_consistent_maps_and_positive_selected_mass() -> None:
    with pytest.raises(ValueError, match="keys must match"):
        DecisionResult("a", {"a": 0.5, "b": 0.5}, {"a": 0, "c": 0}, 0.1)
    with pytest.raises(ValueError, match="softmax"):
        DecisionResult("a", {"a": 0.2, "b": 0.8}, {"a": 0, "b": 0}, 0.1)
    with pytest.raises(ValueError, match="positive probability"):
        DecisionResult("b", {"a": 1, "b": 0}, {"a": 0, "b": -1000}, 0.1)
    with pytest.raises(ValueError, match="action sets"):
        ScoredDecision(
            ScoreResult({"a": 0, "b": 0}, {"a": 0.5, "b": 0.5}, {"a": 1, "b": 2}),
            DecisionResult("a", {"a": 0.5, "c": 0.5}, {"a": 0, "c": 0}, 0.1),
        )


def test_async_scorer_accepts_host_implementation() -> None:
    class HostScorer:
        async def score(self, request: ScoreRequest) -> ScoreResult:
            await asyncio.sleep(0)
            return ScoreResult({"a": 0, "b": 0}, {"a": 0.5, "b": 0.5}, {"a": 1, "b": 2})

    request = ScoreRequest(
        (ChatMessage("user", "Choose"),), ActionSpace.from_mapping({"a": "A", "b": "B"})
    )
    scorer: AsyncActionScorer = HostScorer()
    assert asyncio.run(scorer.score(request)).probabilities == {"a": 0.5, "b": 0.5}
