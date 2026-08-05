"""Calibrated discrete action selection for language-model agents."""

from .correction import (
    CorrectionResult,
    bayesian_correct,
    contextual_log_odds,
    decide,
    expected_action_rates,
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
    "ActionSpace",
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
    "contextual_log_odds",
    "counted_prior",
    "decide",
    "expected_action_rates",
    "fit_marginal_intercepts",
    "pairwise_log_odds",
    "probability_mean_prior",
    "sample_probabilities",
    "softmax",
]

__version__ = "0.1.0"
