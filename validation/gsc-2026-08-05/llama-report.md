# Live backend parity

Model: `unsloth/Llama-3.1-8B-Instruct`

## Score parity

| Comparison | Max log-odds error | Max corrected-probability error | Token mismatches | Pass |
|---|---:|---:|---:|:---:|
| api_vs_vllm_native | 1.71829e-07 | 1.6311e-08 | 0 | yes |
| api_vs_transformers | 0.25 | 0.0550657 | 0 | no |

## Native processor parity

| Backend | Decisions | Max expected/reported error | β=0 target error | Pass |
|---|---:|---:|---:|:---:|
| vllm-native | 24 | 2.616e-07 | 2.44513e-08 | yes |
| transformers | 24 | 3.48263e-07 | 2.44513e-08 | yes |

## Batch-shape diagnostic

Sequential-vs-batched max log-odds error: `0.125`

Required implementation checks pass: **True**

Cross-runtime vLLM/Transformers diagnostic passes its strict tolerance: **False**
