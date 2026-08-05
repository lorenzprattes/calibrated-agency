from __future__ import annotations

from dataclasses import dataclass

from calibrated_agency.conformance import compare_backends
from calibrated_agency.correction import softmax
from calibrated_agency.schemas import ActionSpace, ChatMessage, ScoreRequest, ScoreResult


@dataclass
class StaticBackend:
    scores: dict[str, float]
    token_ids: dict[str, int]

    def score(self, request: ScoreRequest) -> ScoreResult:
        assert set(request.action_space.actions) == set(self.scores)
        return ScoreResult(
            raw_scores=self.scores,
            probabilities=softmax(self.scores),
            token_ids=self.token_ids,
        )


def requests() -> list[ScoreRequest]:
    space = ActionSpace.from_mapping({"a": "A", "b": "B"})
    return [ScoreRequest((ChatMessage("user", "choose"),), space)]


def test_backend_comparison_ignores_additive_score_constant() -> None:
    first = StaticBackend({"a": 1.2, "b": -0.4}, {"a": 10, "b": 11})
    second = StaticBackend({"a": 101.2, "b": 99.6}, {"a": 10, "b": 11})
    report = compare_backends(
        first,
        second,
        requests(),
        prior={"a": 0.4, "b": 0.6},
        target={"a": 0.2, "b": 0.8},
    )
    assert report.passed
    assert report.max_absolute_log_odds_difference < 1e-12
    assert report.max_corrected_probability_difference < 1e-12


def test_backend_comparison_reports_semantic_mismatch() -> None:
    first = StaticBackend({"a": 1.2, "b": -0.4}, {"a": 10, "b": 11})
    second = StaticBackend({"a": 1.3, "b": -0.4}, {"a": 10, "b": 12})
    report = compare_backends(
        first,
        second,
        requests(),
        prior={"a": 0.4, "b": 0.6},
        target={"a": 0.2, "b": 0.8},
        log_odds_tolerance=0.01,
    )
    assert not report.passed
    assert report.token_id_mismatches == 1
    assert report.max_absolute_log_odds_difference > 0.09
