"""Calibrated discrete action selection for language-model agents."""

from .backends.base import ActionScorer, AsyncActionScorer
from .correction import (
    CorrectionResult,
    bayesian_correct,
    bayesian_correct_log,
    contextual_log_odds,
    decide,
    expected_action_rates,
    log_normalize_weights,
    log_softmax,
    pairwise_log_odds,
    sample_probabilities,
    softmax,
)
from .priors import (
    MarginalFit,
    apply_intercepts,
    counted_prior,
    fit_marginal_intercepts,
    probability_mean_prior,
)
from .schemas import (
    ActionSpace,
    Candidate,
    ChatMessage,
    DecisionResult,
    ScoredDecision,
    ScoreRequest,
    ScoreResult,
)

__all__ = [
    "ActionScorer",
    "ActionSpace",
    "AsyncActionScorer",
    "Candidate",
    "ChatMessage",
    "CorrectionResult",
    "DecisionResult",
    "MarginalFit",
    "ScoreRequest",
    "ScoreResult",
    "ScoredDecision",
    "apply_intercepts",
    "bayesian_correct",
    "bayesian_correct_log",
    "contextual_log_odds",
    "counted_prior",
    "decide",
    "expected_action_rates",
    "fit_marginal_intercepts",
    "log_normalize_weights",
    "log_softmax",
    "pairwise_log_odds",
    "probability_mean_prior",
    "sample_probabilities",
    "softmax",
]

__version__ = "0.2.0a1"
