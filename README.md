# calibrated-agency

Bayesian calibration of discrete language-model actions while retaining measurable
contextual variation.

> **Alpha:** the mathematical core is backend-independent. The first tested serving target
> is vLLM 0.26. The vLLM processor interface is version-pinned because upstream marks it as
> evolving.

## What the method does

For action `a` in context `x`, the paper-style correction is

```text
corrected_score(a, x)
  = beta * (raw_score(a, x) - log(model_prior(a)))
    + log(target_rate(a))
```

The corrected scores are normalized and sampled over the declared finite action set.

- `beta = 0` samples the target distribution exactly.
- `beta > 0` retains context-dependent model variation.
- Between any two contexts, each pairwise action log-odds difference is scaled exactly by
  `beta`, provided the prompt, labels, and prior are held fixed.
- For `beta > 0`, the closed form does **not** generally guarantee that the marginal action
  rate remains exactly equal to the target. It is an anchor, not a hard constraint.

The project calls this property **calibrated context-responsiveness**. It does not equate a
logit-level property with philosophical or behavioral autonomy.

## Implementations

- **Core:** dependency-free correction, prior estimation, exact seeded sampling, cache
  fingerprints, and an optional exact marginal-intercept fitter.
- **vLLM native:** a vLLM V1 batch logits processor supporting the paper-style trigger and a
  simpler first-generated-token mode.
- **vLLM API:** exact single-token scoring with the vLLM-specific `allowed_token_ids` and
  `logprob_token_ids` fields against a server configured with raw logprobs.
- **Transformers native:** direct next-token scoring and a first-token Transformers
  `LogitsProcessor`.

These are backend-specific adapters. The project does not claim generic
“OpenAI-compatible” support.

## Install

Core only:

```bash
pip install calibrated-agency
```

From a checkout:

```bash
uv sync --group dev
```

Optional runtimes:

```bash
pip install 'calibrated-agency[transformers]'
pip install 'calibrated-agency[vllm]'
```

The combined environment lets vLLM supply its exact PyTorch build:

```bash
pip install 'calibrated-agency[all]'
```

## Core example

```python
from calibrated_agency import bayesian_correct, decide

raw_scores = {"post": 1.7, "comment": 0.2}
model_prior = {"post": 0.65, "comment": 0.35}
target = {"post": 0.10, "comment": 0.90}

corrected = bayesian_correct(raw_scores, model_prior, target, beta=0.15)
choice = decide(raw_scores, model_prior, target, beta=0.15, seed=7)

print(corrected.probabilities)
print(choice.action)
```

See [`examples/synthetic_binary.py`](examples/synthetic_binary.py) for a reproducible
simulation and exact-marginal comparison.

## Endpoint requirements

The vLLM API fast path requires each action label to be a distinct single token at the
**rendered generation boundary**. It deliberately fails closed when this cannot be verified.
Run the server with an explicit score mode:

```bash
vllm serve MODEL --logprobs-mode raw_logprobs
```

The adapter generates one constrained token, retrieves every requested candidate score by
ID, applies the correction in the client, and samples with a client-side seeded RNG.
Server-selected labels are not used to estimate priors.

## Priors

For endpoint and direct-scoring backends, prefer the probability-mean prior: normalize the
candidate scores for each representative historical context, then average those
probabilities. This uses more information and less sampling noise than counting one sampled
label per context.

The sampled-count estimator with additive smoothing remains available to reproduce the
paper protocol.

A reusable prior must be keyed by the full scoring identity: model and tokenizer revisions,
chat-template and prompt hashes, action labels/order/token IDs, menu, context dataset,
estimator, score mode, backend version, and generation settings.

## Validation

```bash
uv run pytest
uv run pytest --cov
uv run ruff check .
uv run mypy
```

Live parity scripts compare pairwise candidate log odds rather than absolute logits, because
additive constants are irrelevant and kernels may differ numerically. See
[`docs/validation.md`](docs/validation.md).

No historical social-media data is distributed. Examples and unit tests are synthetic.

## Documentation

- [Method, guarantees, and limits](docs/method.md)
- [Calibration and “autonomy”](docs/autonomy.md)
- [Backend contracts](docs/backends.md)
- [Validation protocol](docs/validation.md)
- [References and provenance](docs/references.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
