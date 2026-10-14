# Calibration and context dependence

Matching action rates does not establish that an agent behaves autonomously. The correction
controls action probabilities. Its effect on useful behavior needs separate evaluation.

## Action rates

At beta zero, expected action probabilities equal the normalized target. At positive beta,
the closed form generally misses that target. Fitted intercepts match it on the calibration
sample, but may miss it on new data.

Report expected probabilities alongside sampled frequencies. A finite run has sampling
noise even when its underlying probabilities are correctly calibrated.

## Responses to context

The correction multiplies between-context differences in pairwise action log odds by beta.
For example, beta 0.15 retains 15% of the model's original difference in log odds between
two contexts, provided the prior, target, and action vocabulary stay fixed.

This property says how much variation remains, not whether that variation is useful.
Models can respond to relevant evidence, irrelevant wording, stereotypes, or formatting.
A useful evaluation distinguishes these responses with controlled changes to the context.

Possible measurements include variation in prior-centered log odds, agreement in the
ranking of contexts, and sensitivity to relevant versus irrelevant changes. Aggregate action
rates alone cannot show whether decisions respond sensibly to context.

## Behavior

Coherent goals, persona fidelity, content quality, and behavior over time require their own
measurements. The formula does not guarantee any of them. Describe beta as a control over
context-dependent action probabilities rather than a general measure of autonomy.
