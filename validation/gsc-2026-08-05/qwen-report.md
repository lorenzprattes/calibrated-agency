# Live backend parity

Model: `Qwen/Qwen3.5-4B`

## Score parity

| Comparison | Max log-odds error | Max corrected-probability error | Token mismatches | Pass |
|---|---:|---:|---:|:---:|
| api_vs_vllm_native | 3.25032e-07 | 3.7061e-08 | 0 | yes |
| api_vs_transformers | 0.125 | 0.0177142 | 0 | no |

## Native processor parity

| Backend | Decisions | Max expected/reported error | β=0 target error | Pass |
|---|---:|---:|---:|:---:|
| vllm-native | 24 | 2.14923e-07 | 2.44513e-08 | yes |
| transformers | 24 | 5.24373e-07 | 2.44513e-08 | yes |

Required implementation checks pass: **True**

Cross-runtime vLLM/Transformers diagnostic passes its strict tolerance: **False**
