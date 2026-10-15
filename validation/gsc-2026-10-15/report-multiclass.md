# Live backend parity

Model: `Qwen/Qwen3.5-4B`

## Score parity

| Comparison | Max log-odds error | Max corrected-probability error | Token mismatches | Pass |
|---|---:|---:|---:|:---:|
| api_vs_vllm_native | 7.15256e-07 | 9.66055e-08 | 0 | yes |
| api_vs_transformers | 0.25 | 0.0234605 | 0 | no |

## Native processor parity

| Backend | Decisions | Max expected/reported error | β=0 target error | Pass |
|---|---:|---:|---:|:---:|
| vllm-native | 24 | 2.55697e-07 | 2.38599e-08 | yes |
| transformers | 24 | 2.97498e-07 | 4.89477e-09 | yes |

## Batch-shape diagnostic

Sequential-vs-batched max log-odds error: `0.3125`

Required implementation checks pass: **True**

Cross-runtime vLLM/Transformers diagnostic passes its strict tolerance: **False**
