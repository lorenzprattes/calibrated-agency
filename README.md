# calibrated-agency

`calibrated-agency` adjusts a language model's action scores using an estimated model prior
and target action rates. It implements the Bayesian logit correction from
[Calibrated but Autonomous](https://openreview.net/forum?id=9EZgvMOHXn).

The core needs Python 3.10 or later and has no runtime dependencies. Optional adapters
support vLLM and Transformers. Version `0.2.0a1` is an alpha; it is not published on PyPI.

## Install

From a checkout:

```bash
pip install .
```

For an optional model runtime, use `pip install '.[vllm]'`,
`pip install '.[transformers]'`, or `pip install '.[all]'`. The combined extra lets vLLM
select its required PyTorch build. Runtime versions are pinned in `pyproject.toml`.

## Choose an action

```python
import random

from calibrated_agency import bayesian_correct, decide

scores = {"post": 1.7, "comment": 0.2}
prior = {"post": 0.65, "comment": 0.35}
target = {"post": 0.10, "comment": 0.90}
rng = random.Random(7)

probabilities = bayesian_correct(scores, prior, target, beta=0.15).probabilities
choice = decide(scores, prior, target, beta=0.15, rng=rng)

print(probabilities)
print(choice.action)
```

Keep the same `rng` across decisions to continue its random sequence. For a single
reproducible call, use `seed=7` instead. Supplying both arguments is an error.

## The correction

For action `a` in context `x`:

```text
score(a, x) = beta * (raw_score(a, x) - log(prior(a))) + log(target(a))
probability(a | x) = softmax(score(a, x))
```

`beta` must lie between zero and one. At zero, the action probabilities equal the
normalized target in every context. Positive beta retains some of the model's response to
context: differences in pairwise log odds between contexts are multiplied by beta.

At positive beta, average action probabilities generally differ from the target.
`fit_marginal_intercepts` is a separate option that matches the target on a supplied
calibration sample. That match need not hold on new contexts.

Use `bayesian_correct_log` when priors and targets are already logged. It accepts finite
log weights without converting them to probabilities first. Priors and targets must be
positive; the application decides how to handle zero target rates.

The [synthetic example](examples/synthetic_binary.py) compares the two correction methods:

```bash
uv run python examples/synthetic_binary.py
```

## Model adapters

| Adapter | What it does |
|---|---|
| vLLM API | Retrieves all candidate scores, then corrects and samples in Python |
| vLLM native | Applies correction in a vLLM V1 logits processor |
| Transformers native | Scores the next token directly or applies a first-token processor |

Each action label must encode to one distinct token and decode exactly to the same label.
Multi-token labels are not supported. The HTTP adapter needs specific vLLM features;
a generic OpenAI-compatible endpoint is insufficient.

For the API adapter, start vLLM with:

```bash
vllm serve MODEL --logprobs-mode raw_logprobs --generation-config vllm
```

The adapter checks tokenization against the prompt IDs and counts returned by generation.
It requires `/tokenize`, `/detokenize`, and assistant-message continuation, and disables
thinking at the action boundary. See [backend setup](docs/backends.md) for the full contract.

Applications can supply their own scorer through `ActionScorer` or `AsyncActionScorer`.
These interfaces use the same request and result types. The application keeps control of
context construction, timeouts, retries, caching, and action execution.

## Calibration

Estimate the prior from contexts that represent the population you intend to simulate.
`probability_mean_prior` averages the full action distributions. `counted_prior` estimates
frequencies from sampled actions with additive smoothing, matching the original paper's
protocol.

Reusing a prior requires the same model, tokenizer, prompt, action vocabulary, and scoring
settings. The package provides fingerprint helpers; it does not manage a persistent cache.
See [the method](docs/method.md) for numerical details and assumptions.

## Development and tests

```bash
uv sync --group dev
uv run pytest --cov
uv run ruff check .
uv run mypy
uv build
```

The current package passed 104 tests on Python 3.10, 3.12, and 3.14, with 93.44% core
branch coverage. The wheel was also tested in a fresh environment.

[Earlier GPU runs](validation/gsc-2026-08-05/README.md) tested version 0.1.0 on Qwen3.5-4B
and Llama-3.1-8B. vLLM API/native scores agreed within the chosen tolerances;
vLLM/Transformers scores did not. Those results do not validate the updated adapters in
0.2.0a1. Fresh model tests remain pending.

See [validation](docs/validation.md), [CPU results](validation/cpu-2026-10-14/README.md),
and [DeSiRe integration](docs/project-alignment.md) for the tested scope and remaining work.
Examples and tests use synthetic data.

## Further reading

- [Method and numerical behavior](docs/method.md)
- [Calibration and context dependence](docs/autonomy.md)
- [Backend setup](docs/backends.md)
- [References](docs/references.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
