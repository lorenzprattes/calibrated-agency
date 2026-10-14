# Method

Let `r_a(x)` be the model's score for action `a` in context `x`, `p0(a)` its estimated
marginal action prior, and `pi(a)` the target distribution. Scores may be raw logits or log
probabilities; adding the same constant to every action does not affect the result.

The correction is:

```text
s_a(x) = beta * (r_a(x) - log p0(a)) + log pi(a)
q(a | x) = softmax_a(s(x))
```

Prior and target weights must be positive. Their normalization is handled by the package.
Use the same prompt construction, action labels, and scoring settings when estimating a
prior and applying it to new contexts.

## Beta and context

At beta zero, `q(a | x) = pi(a)` in every context. With a fixed prior and target, the
between-context difference in pairwise log odds satisfies:

```text
log(q(a|x) / q(b|x)) - log(q(a|y) / q(b|y))
  = beta * ((r_a(x) - r_b(x)) - (r_a(y) - r_b(y)))
```

The prior and target cancel in this difference. Positive beta preserves the ordering of
contexts for each action pair, while reducing the size of the difference. Beta zero removes
that context variation entirely.

For positive beta, the average of `q` over contexts generally differs from `pi`. Softmax
normalizes each context separately, so averaging corrected probabilities does not reverse
the correction's normalization. Treat the target as a reference distribution unless you
have measured the resulting marginal rates.

The formula preserves variation regardless of its cause. Relevant evidence, prompt wording,
and model biases can all affect the scores. The correction alone cannot distinguish them.

## Estimating the prior

`probability_mean_prior` averages action distributions from representative contexts. Obtain
all candidate scores, normalize them within each context, then average the probabilities.
This uses all available scores and avoids drawing an action just to estimate its frequency.

`counted_prior` instead counts sampled actions and adds a pseudocount to each action before
normalizing. With add-one smoothing, it reproduces the original paper's prior estimator.

The calibration contexts should represent the population for which the prior is intended.
Estimating it from only the immediate decision context can remove the very signal you want
to preserve.

## Fitted intercepts

`fit_marginal_intercepts` fits a different correction:

```text
q(a | x) = softmax_a(beta * r_a(x) + intercept_a)
```

It adjusts a constant intercept for each action until average probabilities on the supplied
calibration set match the target within the requested tolerance. Constant intercepts preserve
the same beta-scaled difference in log odds between contexts.

This is a finite-sample fit, not the paper's closed form. A new context distribution or a
changed action menu can move later rates away from the fitted target.

## Temperature

The supported native processors run before the runtime divides logits by temperature `T`.
Writing corrected logits directly would produce `softmax(log pi / T)` at beta zero, rather
than `pi`. The processors multiply corrected logits by `T` to compensate for that division.

The API adapter samples corrected probabilities in Python. Text continuation has a separate
temperature. Native decision methods require positive temperature; greedy selection cannot
sample the requested categorical distribution.

## Numerical behavior

`bayesian_correct` normalizes positive weights in log space. `bayesian_correct_log` accepts
finite log weights directly, including values too small to represent as probabilities.
Negative infinity is rejected. An application that permits zero target rates must decide
whether to exclude those actions or apply an explicit positive floor.

`log_softmax` retains log probabilities when exponentiation would round them to zero.
Returned probability maps may therefore contain zeros. Sampling and rate averaging accept
these maps, but reject negative, nonfinite, or all-zero weights.

`probability_mean_prior` allows an action to have zero mass in an individual context. If it
has zero mass across the whole calibration set, a positive pseudocount is required. The
function does not add one implicitly. DeSiRe uses `1e-12` for this estimator.

Fitted-intercept updates also run in log space. Convergence is measured by absolute
probability error, not relative error for rare actions. The fitter raises an error if it
cannot converge within the iteration limit. Unrepresentable score ranges also raise an
error rather than substituting another distribution.
