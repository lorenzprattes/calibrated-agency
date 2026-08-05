# Calibration and “autonomy”

“Autonomy” is too broad to infer from an action-rate metric. This project separates three
claims.

## 1. Marginal calibration

Does the expected frequency of each action match a declared target?

- The closed form guarantees this at beta zero.
- At positive beta it usually approximates rather than guarantees the target.
- Fitted marginal intercepts guarantee it on the calibration-context sample.

Report expected probabilities as well as sampled frequencies. A finite run adds sampling
noise that should not be mistaken for calibration error.

## 2. Context-responsiveness

Does the agent alter its action propensity across contexts?

The closed form retains a beta fraction of the model's between-context pairwise log-odds
variation. This is a mathematical property, not a judgment that the variation is useful.
At beta 0.15, for example, only 15% of that log-odds variation remains.

Useful measurements include:

- variance of prior-centered pairwise log odds;
- rank correlation across contexts;
- sign agreement of contextual deviations;
- sensitivity to controlled, causally relevant context changes;
- insensitivity to paraphrases and irrelevant changes.

Comparing only final marginal action rates cannot establish context-responsiveness. A strong
target anchor can make two behaviorally different scoring protocols look similar in the
aggregate.

## 3. Behavioral agency

Does the generated action serve coherent goals, respond to relevant evidence, and remain
stable under harmless prompt changes?

That requires task-specific evaluation. The correction can preserve model variation caused
by prompt artifacts or bias just as readily as variation caused by sound reasoning. Content
quality, long-horizon planning, user-specific fidelity, and downstream consequences are
outside the formula's guarantee.

## Recommended language

Prefer:

> The policy calibrates discrete action rates while retaining a controlled amount of the
> model's context-dependent action signal.

Avoid:

> The policy guarantees autonomous behavior.

A beta value is a transparent control over one operational dimension of agency. It is not a
scalar measurement of autonomy in general.
