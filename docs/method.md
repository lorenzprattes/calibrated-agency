# Method, guarantees, and limits

## Closed-form correction

Let `r_a(x)` be a model score for action `a` in context `x`. It may be a raw logit or a raw
log-probability; any additive constant shared by all actions cancels. Let `p0(a)` be the
model's marginal action prior on representative contexts, and let `pi(a)` be the desired
action-rate anchor.

The method defines

```text
s_a(x) = beta * (r_a(x) - log p0(a)) + log pi(a)
q(a|x) = softmax_a(s(x))
```

All prior and target entries must be strictly positive. Candidate labels and the context in
which they are scored must stay fixed between prior estimation and use.

## What is exact

At `beta=0`, `q(a|x)=pi(a)` for every context. The marginal therefore equals the target
exactly.

For any actions `a,b` and contexts `x,y`,

```text
[log q(a|x)/q(b|x) - log q(a|y)/q(b|y)]
  = beta *
    [logit_score(a,b,x) - logit_score(a,b,y)]
```

The target and prior terms cancel because they are constant across contexts. Consequently:

- positive beta preserves each pairwise ordering of contexts;
- beta scales the magnitude of every between-context pairwise log-odds difference;
- beta zero removes all model context variation from the action decision.

This is the project's operational context-responsiveness guarantee.

## What is not exact

For positive beta, averaging `q(a|x)` over contexts does not generally recover `pi(a)`.
Softmax normalizes separately in each context, so the average of normalized ratios is not
the normalized average. The closed form is exact under beta zero and in some special cases,
but otherwise the target is an anchor.

The method also does not establish that model context sensitivity is desirable. It can
preserve sensitivity to artifacts, stereotypes, prompt wording, or irrelevant features.
Behavioral validation remains necessary.

## Prior estimation

### Probability mean (recommended)

For each representative context, obtain every candidate's raw score, normalize over the
candidate set, and average the resulting probabilities. This is deterministic given the
scores and uses all information returned by the model.

The contexts should represent the marginal population from which `p0` is meant to remove a
baseline tendency. Reusing the immediate decision context as the prior population can
subtract the signal one intends to preserve.

### Sampled counts (reproduction mode)

Sample one constrained action per prior context, count actions, and apply additive smoothing.
This reproduces the original experimental protocol but introduces avoidable Monte Carlo
variance when scores are available directly.

## Exact marginal intercepts

`fit_marginal_intercepts` offers a separate finite-sample calibration mode:

```text
q(a|x) = softmax_a(beta * r_a(x) + intercept_a)
```

It iteratively chooses constant action intercepts until the average distribution on supplied
calibration contexts equals the target. Because intercepts are constant over contexts, the
same pairwise context-responsiveness identity holds.

This guarantee is empirical: it applies to the calibration contexts. Distribution shift,
endogenous simulation dynamics, or a changed action menu can move future rates away from the
target.

## Temperature

Both vLLM and Transformers apply ordinary generation temperature after custom logits
processors. If corrected logits are written directly and then divided by temperature `T`,
the beta-zero distribution becomes `softmax(log pi / T)` rather than `pi`.

The native processors therefore multiply corrected action logits by `T` before the built-in
temperature step. The API path retrieves pre-temperature raw scores and samples the corrected
distribution client-side. Text continuation temperature is separate from action-selection
temperature.
