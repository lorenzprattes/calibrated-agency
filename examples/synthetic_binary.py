"""Synthetic calibration/context-responsiveness demonstration.

Run with:
    uv run python examples/synthetic_binary.py
"""

from __future__ import annotations

import random

from calibrated_agency import (
    apply_intercepts,
    bayesian_correct,
    expected_action_rates,
    fit_marginal_intercepts,
    probability_mean_prior,
    softmax,
)


def main() -> None:
    rng = random.Random(20260805)
    raw_context_scores = [
        {"act": rng.gauss(-0.2, 1.4) + context_effect, "wait": 0.0}
        for context_effect in ([1.0] * 100 + [-0.5] * 300)
    ]
    model_distributions = [softmax(scores) for scores in raw_context_scores]
    model_prior = probability_mean_prior(model_distributions)
    target = {"act": 0.10, "wait": 0.90}

    print(f"model prior: {model_prior}")
    print(f"target:      {target}\n")
    print("paper closed form")
    print("beta  expected P(act)  retained log-odds variation")
    for beta in (0.0, 0.15, 0.5, 1.0):
        corrected = [
            bayesian_correct(scores, model_prior, target, beta).probabilities
            for scores in raw_context_scores
        ]
        rate = expected_action_rates(corrected)["act"]
        print(f"{beta:>4.2f}  {rate:>15.4f}  {beta:>27.0%}")

    beta = 0.5
    fitted = fit_marginal_intercepts(raw_context_scores, target, beta)
    exact = [
        apply_intercepts(scores, fitted.intercepts, beta) for scores in raw_context_scores
    ]
    print("\nexact intercept fit on these calibration contexts")
    print(f"beta:              {beta}")
    print(f"expected P(act):   {expected_action_rates(exact)['act']:.12f}")
    print(f"iterations:        {fitted.iterations}")
    print(f"maximum error:     {fitted.max_absolute_error:.3g}")
    print("\nThe exact fit retains 50% of between-context log-odds variation, but its")
    print("target guarantee applies to the fitted context distribution, not arbitrary future data.")


if __name__ == "__main__":
    main()
