# GSC A100 live validation — 2026-08-05

Repository commit: `c2fbe2a`  
Runtime: vLLM 0.26.0, Transformers 5.14.1, PyTorch 2.11.0+cu130  
Hardware: two NVIDIA A100-PCIE-40GB GPUs, driver 580.65.06  
Execution: bfloat16; vLLM eager mode; synthetic prompts only

| Model | Immutable model revision | Slurm job |
|---|---|---:|
| Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 5563540 |
| unsloth/Llama-3.1-8B-Instruct | `4699cc75b550f9c6f3173fb80f4703b62d946aa5` | 5563568 |

## Main result

The vLLM HTTP and in-process implementations agree to far below the predeclared tolerance:

| Model | Max API/native log-odds error | Max corrected-probability error |
|---|---:|---:|
| Qwen3.5-4B | 3.25e-7 | 3.71e-8 |
| Llama-3.1-8B | 1.72e-7 | 1.63e-8 |

All candidate token IDs agreed. This establishes implementation parity for the tested
single-token, first-position vLLM profile.

Both native processors also reproduced the shared core's corrected probabilities at action
temperatures 0.7 and 1.0:

| Model/backend | Max expected/reported error | β=0 target error |
|---|---:|---:|
| Qwen / vLLM native | 2.15e-7 | 2.45e-8 |
| Qwen / Transformers | 5.24e-7 | 2.45e-8 |
| Llama / vLLM native | 2.62e-7 | 2.45e-8 |
| Llama / Transformers | 3.48e-7 | 2.45e-8 |

This directly checks the temperature compensation that is absent from a naive pre-temperature
logits processor.

## Deliberate strict diagnostic failure

The required implementation checks pass. The separate cross-runtime diagnostic is `False`
because raw next-token scores from vLLM and Transformers did not meet its predeclared strict
tolerance:

| Model | Max vLLM/Transformers log-odds error | Max corrected-probability error |
|---|---:|---:|
| Qwen3.5-4B | 0.125 | 0.0177 |
| Llama-3.1-8B | 0.250 | 0.0551 |

Token IDs still agreed, and each Transformers processor matched the correction computed from
the scores it received. The discrepancy therefore concerns the underlying bfloat16 model
execution across different runtime implementations/kernels, not the correction formula.
Users should not expect vLLM and Transformers to produce identical trajectories merely
because they use the same checkpoint.

At beta 0.15, cross-runtime maximum corrected-probability differences were much smaller:
0.0041 for Qwen and 0.0064 for Llama. The strict report includes beta 1, where runtime score
differences pass through unattenuated.

## Diagnostic findings retained from development

- vLLM's batch-invariance mode cannot run Qwen3.5's GDN attention in vLLM 0.26.
- A non-eager Qwen run produced a 0.125 API/native log-odds discrepancy across independent
  engine processes. Eager execution reduced it below 3.3e-7. This is why execution mode is
  part of provenance and why the supplied validation script uses eager mode.
- The Llama checkpoint's generation config supplied temperature 0.6 and top-p 0.9. The final
  deployment uses `--generation-config vllm`, and both API and Transformers action requests
  explicitly neutralize inherited truncation filters.
- Transformers 5 returned a `BatchEncoding` by default for Qwen's chat template. Candidate
  validation now explicitly requests a flat token list.

## Scope

Eight synthetic contexts used the `post` and `comment` labels after an exact `<action>`
assistant prefix. The run checked score parity, boundary token IDs, beta values 0, 0.15, 0.5,
and 1, and native processor distributions at temperatures 0.7 and 1.0. It did not test
multi-token labels, beams, speculative decoding, production CUDA-graph mode, long-form
continuation quality, or behavioral usefulness of the model's contextual signal.

Machine-readable summaries and their hashes are committed beside this file. Full raw logs and
per-context artifacts remain under the ignored local `results/` directory.
