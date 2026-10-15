# Qwen3.5-4B backend checks — 15 October 2026

The updated `0.2.0a1` adapters passed the correction checks. API and native vLLM agreed
within the declared tolerances. Transformers and vLLM still produced different underlying
model scores. A follow-up traced both adapters against direct runtime calls: neither adapter
introduced the difference on these inputs.

These are correctness benchmarks on synthetic prompts. They do not measure behavioral
quality, held-out calibration, or inference throughput.

## Setup

- Model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Runtime: vLLM 0.26.0, Transformers 5.14.1, PyTorch 2.11.0+cu130; NVIDIA H200.
- Weights and inference used bfloat16. vLLM ran eagerly, without CUDA graphs. Transformers
  used SDPA attention and its PyTorch fallback for Qwen3.5's gated delta layers; the optional
  fast-path dependencies were unavailable. vLLM selected FlashAttention 3 for full attention.
- Eight forum contexts for each of two action sets: two actions and eleven actions.
  All action labels were distinct single tokens with exact decoding. The assistant prefix
  was `<action>`, with thinking disabled.
- Betas: 0, 0.15, 0.5, 1. Native processor checks also used action temperatures 0.7 and 1.
  Each backend/case had 24 native decisions: three contexts, four betas, two temperatures.

The [plan](plan.json) retains the original source hashes and thresholds. The source snapshot
was based on commit `d697fc26d4852b183d4347b73668501fe464b1d0` plus the hashed study changes.
[CPU preflight](cpu-preflight.json) checked cached inputs, tokenization, prompt lengths, and
both native processors before GPU allocation. Its largest probability error was `4.04e-8`.

## Initial comparison

| Check | Two actions | Eleven actions | Threshold |
|---|---:|---:|---:|
| API/native maximum log-odds error | 7.45e-8 | 7.15e-7 | 0.001 |
| API/native maximum corrected-probability error | 1.48e-8 | 9.66e-8 | 0.0001 |
| vLLM processor maximum probability error | 3.69e-7 | 2.56e-7 | 0.00001 |
| Transformers processor maximum probability error | 1.99e-7 | 2.97e-7 | 0.00001 |
| Largest beta-zero target error across processors | 2.45e-8 | 2.39e-8 | 0.0000001 |

All required checks passed. Candidate token IDs agreed across the three backends.

The separate cross-runtime diagnostic failed its unchanged tolerances:

| API vs Transformers | Two actions | Eleven actions | Threshold |
|---|---:|---:|---:|
| Maximum log-odds error | 0.1250001 | 0.2500005 | 0.05 |
| Maximum corrected-probability error | 0.03103 | 0.02346 | 0.01 |

Probability errors are absolute differences: `0.03103` means about 3.1 percentage points.
The reports include beta 1, where the correction retains score differences without attenuation.
The maximum sequential/batched vLLM log-odds drift was 0.125 for two actions and 0.3125
for eleven actions. Batch shape therefore matters even within one runtime.

See the [binary summary](summary-binary.json), [multiclass summary](summary-multiclass.json),
and per-context `api-*`, `vllm-native-*`, and `transformers-*` JSON files beside this record.

## Follow-up: bypass the adapters

The follow-up used all 16 prompts and the same immutable model revision. It checked actual
prompt token IDs against those returned by the original API generation, along with action IDs.
All matched. Each diagnostic backend ran in its own process on the same H200 allocation.
Native vLLM returned processed logits, used neutral generation defaults, and disabled prefix
caching to remove cache reuse from the comparison. Its direct call omitted our processor's
request configuration. Transformers used the same loaded model for every controlled variation.

| Controlled comparison | Maximum candidate-score difference | Maximum log-odds difference |
|---|---:|---:|
| Transformers adapter vs direct forward | 0 | 0 |
| Transformers repeated request | 0 | 0 |
| Transformers last-token-only projection vs full projection | 0 | 0 |
| Transformers caching disabled vs default | 0 | 0 |
| vLLM adapter/neutral processor vs direct runtime call | 0 | 0 |
| vLLM repeated request | 0 | 0 |
| Transformers eager attention vs SDPA | 0.125 | 0.25 |
| vLLM eight-request batch vs sequential | 0.25 | 0.3125 |

The vLLM runtime also reported exactly the prompt IDs supplied by the diagnostic. These checks
rule out prompt/token selection, candidate-score extraction, the neutral processor, and the
Transformers cache/projection settings tested here as explanations for the discrepancy.

A float32 projection of the same Transformers hidden state and candidate weights changed
individual scores by at most 0.06230. Rounding that projection back to bfloat16 reproduced
every original candidate score exactly. This isolates output rounding, but does not make
float32 projection a fix: maximum cross-runtime log-odds error remained 0.23518, compared
with 0.25 for the normal adapter. Switching Transformers to eager attention increased that
cross-runtime maximum to 0.375.

The evidence supports numerical differences inside model execution, including attention and
batch-shape effects. It does not identify one faulty upstream kernel or establish which
runtime is closer to a high-precision reference. The follow-up did not run the entire model
in float32 or test other checkpoints. No production adapter change was justified by this run.

The [diagnostic summary](diagnostic-summary.json) and `diagnostic-*.json` traces retain the
controlled comparisons. Use a prior calibrated with the same runtime and execution profile
as the decisions; do not assume checkpoint identity implies identical action probabilities.

## Reproduce the analysis

From an installed development checkout, the saved results need no GPU:

```bash
python -m scripts.compare_live \
  --api validation/gsc-2026-10-15/api-binary.json \
  --vllm-native validation/gsc-2026-10-15/vllm-native-binary.json \
  --transformers validation/gsc-2026-10-15/transformers-binary.json \
  --output /tmp/binary-summary.json --report /tmp/binary-report.md
python -m scripts.validation_study.summarize_diagnostic \
  --root validation/gsc-2026-10-15
```

Repeat the first command with `multiclass` to reproduce that case. `SHA256SUMS` covers the
retained artifacts and executed diagnostic source. The source files are under
`scripts/validation_study/`; `diagnose.py` performs the controlled runtime calls.

To repeat model execution, cache the pinned model revision and install the runtime versions
above. Copy `plan.json` to a writable run directory and point `HF_HUB_CACHE` to the cache.
Run `python -m scripts.validation_study.preflight --root RUN` on CPU first.
`worker --root RUN --mode MODE` runs both cases; API mode also needs `--base-url URL`.
Start its server with the pinned revision, `--dtype bfloat16 --enforce-eager`,
`--max-model-len 4096 --generation-config vllm --logprobs-mode raw_logprobs`.
After the workers finish, `diagnose --root RUN --mode transformers` and
`diagnose --root RUN --mode vllm-native` reproduce the follow-up in separate processes.
The diagnostic expects the initial API results under `RUN/results/api/`.
Supply your own scheduler/container paths if using the optional GSC launch helpers.

## Compute and operational limits

Initial Slurm array: `5736565`, all three tasks completed with exit zero. The three independent
one-GPU tasks used 156, 144, and 41 seconds of allocation, totaling 5 minutes 41 seconds.
Two tasks ran concurrently; the third reused a released GPU. Follow-up job `5736889` completed
with exit zero in 2 minutes 53 seconds. Total allocated GPU time was 8 minutes 34 seconds,
including startup. These timings are not a controlled speed comparison.

The original completion monitor failed when Slurm removed the finished array from the live
queue. That also delayed later agent audits and result collection. Accounting confirmed normal
job completion; the separate in-job supervisors remained in place. Completion handling now
consults accounting for an absent job, preserves probe errors, and lets due agent audits run
when a probe fails. Both fixes have CPU regression checks. No automatic resubmission occurred.

The follow-up emitted a Python multiprocessing semaphore-cleanup warning at vLLM shutdown.
All 16 results were written and the allocation completed normally; this warning is retained
as an operational limitation rather than counted as a score failure.

Public artifacts retain synthetic requests, scores, seeds, token IDs, versions, and source
hashes. Host paths and full HTTP response envelopes are omitted. Full local logs remain outside
the repository. The record does not include the GPU driver version or a container image digest.
