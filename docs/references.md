# References and implementation sources

## Method provenance

This package is a fresh, generalized implementation of the Bayesian action-prior correction
used in the following research codebase:

- https://github.com/cs2-lab/bayesian-inference-gabm
- reference revision inspected during development:
  `def1eaa6508f21bbaf0ac5fee2147c6dcd201d25`

A formal paper citation has not yet been added because publication metadata was not included
in the inspected repository. Add the definitive citation before a public release rather than
inventing or guessing bibliographic fields.

## Runtime contracts

The implementation and validation protocol were checked against version-tagged official
sources:

- vLLM 0.26 custom logits processors:
  https://docs.vllm.ai/en/v0.26.0/features/custom_logitsprocs/
- vLLM 0.26 sampling parameters:
  https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/sampling_params.py
- vLLM 0.26 sampler ordering:
  https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/v1/sample/sampler.py
- vLLM 0.26 chat protocol:
  https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/entrypoints/openai/chat_completion/protocol.py
- Transformers 5.14.1 generation:
  https://raw.githubusercontent.com/huggingface/transformers/v5.14.1/src/transformers/generation/utils.py
- Transformers 5.14.1 logits processors:
  https://raw.githubusercontent.com/huggingface/transformers/v5.14.1/src/transformers/generation/logits_process.py
- PyTorch reproducibility:
  https://docs.pytorch.org/docs/stable/notes/randomness.html

The vLLM HTTP protocol source exposes `logprob_token_ids` in 0.26 even though some generated
SamplingParams prose still describes it as engine-only. The live endpoint integration test
is retained because of that documentation inconsistency.
