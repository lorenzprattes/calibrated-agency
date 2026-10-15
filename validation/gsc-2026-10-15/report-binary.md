# Live backend parity

Model: `Qwen/Qwen3.5-4B`

## Score parity

| Comparison | Max log-odds error | Max corrected-probability error | Token mismatches | Pass |
|---|---:|---:|---:|:---:|
| api_vs_vllm_native | 7.45058e-08 | 1.48195e-08 | 0 | yes |
| api_vs_transformers | 0.125 | 0.0310304 | 0 | no |

## Native processor parity

| Backend | Decisions | Max expected/reported error | β=0 target error | Pass |
|---|---:|---:|---:|:---:|
| vllm-native | 24 | 3.69313e-07 | 2.44513e-08 | yes |
| transformers | 24 | 1.9896e-07 | 2.44513e-08 | yes |

## Batch-shape diagnostic

Sequential-vs-batched max log-odds error: `0.125`

Required implementation checks pass: **True**

Cross-runtime vLLM/Transformers diagnostic passes its strict tolerance: **False**
