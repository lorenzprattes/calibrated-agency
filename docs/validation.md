# Validation protocol

Validation has three layers. Passing one layer does not substitute for the others.

## 1. Mathematical and unit tests

The dependency-free suite checks:

- beta-zero identity;
- the closed-form pairwise log-odds equation;
- invariance to additive score constants;
- exact beta scaling of between-context log-odds differences;
- deterministic client-side sampling;
- probability-mean and sampled-count priors;
- convergence and calibration of fitted marginal intercepts;
- cache fingerprint sensitivity;
- strict action/key/finite-value validation;
- vLLM HTTP payload and response parsing through a fake endpoint;
- native processor configuration and activation state.

Core coverage excludes optional backend modules because importing and executing their main
paths requires the corresponding runtime. Those modules are covered by GPU integration
runs.

## 2. Live backend conformance

For each model, render the same synthetic chat messages and assistant prefix through each
backend. The supplied GSC script uses eager vLLM execution to reduce CUDA-graph differences
between the server and offline engine; production deployments should separately test their
actual execution mode. Verify before scoring that:

- every label appends exactly one token;
- token IDs agree across vLLM API, vLLM native, and Transformers;
- all requested scores are finite and present;
- no unlisted token can be selected;
- the deployed vLLM score mode is explicit.

Compare pairwise candidate log odds, not absolute logits. An additive constant has no effect
on the categorical policy.

Predeclared numerical thresholds:

| Comparison | Role | Max pairwise log-odds error | Max corrected-probability error |
|---|---|---:|---:|
| vLLM API vs vLLM native | required implementation gate | 0.001 | 0.0001 |
| vLLM API vs Transformers | strict cross-runtime diagnostic | 0.05 | 0.01 |

The Transformers diagnostic is not a release gate: vLLM and Transformers use different
model implementations and kernels, so they need not expose identical bfloat16 logits. Each
native processor must instead match the shared correction applied to the scores from its own
invocation. The diagnostic remains visible and must not be relabeled as passing when it
exceeds its threshold.

The native vLLM run also scores all contexts in one scheduler batch and reports the drift
against sequential scoring. This is a diagnostic rather than an implementation gate because
vLLM 0.26 does not provide batch invariance for every architecture (notably Qwen3.5 GDN
attention). Deployments that batch decisions should calibrate and validate under that same
batching profile.

The Transformers tolerance is wider because different kernels and execution paths are not
expected to be bitwise identical.

For beta values 0, 0.15, 0.5, and 1:

- compare corrected probabilities from the shared core;
- compare each native processor's backend-reported processed distribution with the shared
  core;
- require beta-zero target error below `1e-7` at action temperatures 0.7 and 1.0;
- repeat fixed requests to expose batch-dependent drift;
- retain raw requests, responses, token IDs, versions, hardware, and seeds.

Continuation is a separate conformance surface. Reject output that terminates by length,
echoes substantial prompt material, emits a contradictory action, or fails the example's
syntax—not merely output that lacks one closing tag.

## 3. Behavioral evaluation

Backend parity proves that implementations realize the same score transformation. It does
not prove that the retained context signal is useful.

A behavioral study should measure:

- marginal calibration on held-out contexts;
- expected rates and sampled rates separately;
- prior-centered log-odds variance;
- contextual rank/sign agreement against an uncorrected reference;
- response to controlled relevant interventions;
- robustness to irrelevant and paraphrastic changes;
- content quality and long-horizon effects.

The closed-form correction and exact-intercept mode must be reported separately.

## Reproducibility record

A live run records at least:

- repository commit and dirty status;
- model and tokenizer IDs and immutable revisions;
- vLLM, Transformers, PyTorch, CUDA, and driver versions;
- model dtype, quantization, attention backend, and parallelism;
- GPU type/count and batch-invariance setting;
- chat template and kwargs;
- action labels and boundary token IDs;
- raw/processed score mode;
- every decoding parameter and seed;
- input and output SHA-256 hashes.

Small numeric differences can remain across hardware and software releases. Claims should use
tolerances and distributions rather than promise universal transcript identity.
