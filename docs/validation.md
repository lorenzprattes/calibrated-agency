# Testing and validation

The unit tests check the correction and adapter code. Model tests check runtime behavior.
A behavioral study is needed to tell whether the resulting actions are useful. These answer
different questions.

## Package tests

Run:

```bash
uv run pytest --cov
uv run ruff check .
uv run mypy
uv build
```

The tests cover beta-zero targets, log-odds identities, additive score shifts, prior estimators,
fitted intercepts, numerical edge cases, random-generator state, and result validation.
Mock endpoint tests check request fields, malformed responses, prompt token checks, and
independence from server response ordering. Native processor tests cover configuration,
activation, and masking without loading a model.

Core coverage excludes the optional backend modules. Mocked backend tests do run, but the
coverage percentage does not measure those modules or establish live runtime support.
The current CPU results are in [the validation record](../validation/cpu-2026-10-14/README.md).

## Model tests

Use the same synthetic messages, action labels, and assistant prefix for each backend.
Check that labels encode to distinct single tokens and decode exactly. Candidate token IDs
must agree, every candidate score must be finite and present, and API generation must return
the same prompt IDs and counts as tokenization.

Compare pairwise action log odds rather than absolute scores: adding a constant to every
score has no effect on action probabilities. The retained model tests use these thresholds:

| Comparison | Purpose | Maximum log-odds error | Maximum corrected-probability error |
|---|---|---:|---:|
| vLLM API vs vLLM native | Implementation comparison | 0.001 | 0.0001 |
| vLLM API vs Transformers | Cross-runtime comparison | 0.05 | 0.01 |

The two runtimes use different model code and kernels. A cross-runtime difference is worth
reporting even when each processor correctly transforms its own input scores. The historical
vLLM/Transformers comparison exceeded its thresholds; it is recorded as a failure.

For each native processor, compare its probabilities with the shared correction applied to
scores from the same invocation where possible. vLLM 0.26 does not expose raw and processed
scores together, so the historical vLLM check used a separate nearby scoring request.

The existing scripts test beta values 0, 0.15, 0.5, and 1, with action temperatures 0.7 and 1.0.
Their beta-zero target-error threshold is `1e-7`. Repeat fixed requests and compare sequential
with batched scoring. Calibrate under the batching profile that will be used for decisions.

The GSC scripts use eager vLLM execution. Results from that setup do not cover another
execution mode automatically. Qwen3.5 GDN attention does not support vLLM 0.26's
batch-invariance mode, and the retained runs show batch-dependent score drift.

Check text continuation separately. Inspect length termination, prompt copying, contradictory
actions, and whether the full response parses correctly. One-token scoring normally ends by
length; that is expected and differs from truncated text continuation.

## Behavioral studies

Measure calibration on held-out contexts, not just the fitted sample. Report expected rates
and sampled rates separately. Controlled changes to relevant evidence, irrelevant wording,
and paraphrases can show which context differences affect the model.

Content quality, persona fidelity, copying, and behavior over time need measurements beyond
action rates. Report the closed form and fitted-intercept results separately.

## Records to keep

For a model run, record the code revision and working changes; model, tokenizer, and template
revisions; runtime and driver versions; hardware; dtype and quantization; attention and
parallelism settings; score mode; action labels and token IDs; decoding parameters; batching;
and seeds. Keep the requests, responses, and input/output hashes needed to repeat it.

The [historical GPU results](../validation/gsc-2026-08-05/README.md) cover version 0.1.0.
The [current H200 benchmarks](../validation/gsc-2026-10-15/README.md) cover the updated
0.2.0a1 adapters and trace their scores against direct runtime calls. Correction checks pass;
cross-runtime and batch-dependent score differences remain.

## Source comparisons

`scripts/check_project_alignment.py` compares selected GABM source functions and DeSiRe's
math module with the package. It checks binary and multiclass correction, priors, fitted
probabilities, and seeded choices. The reports include code hashes, revisions, interpreter,
seed, and tolerance.

That check uses simulated tensor access and does not execute a model, GPU scheduler, or
simulation. The saved reports describe the initial alpha snapshot. Further harness integration
checks remain deferred; see [DeSiRe integration](project-alignment.md).
