# References

## Paper and original implementation

Ruggero Marino Lazzaroni, Lorenz Prattes, and Jana Lasser.
*Calibrated but Autonomous: Inference-Time Bayesian Logit Correction for LLM Social Simulations.*
2026. [OpenReview](https://openreview.net/forum?id=9EZgvMOHXn),
[PDF](https://openreview.net/pdf?id=9EZgvMOHXn).

The package generalizes the correction from the
[Reddit GABM implementation](https://github.com/cs2-lab/bayesian-inference-gabm).
The inspected revision is `def1eaa6508f21bbaf0ac5fee2147c6dcd201d25`.

## Runtime sources

The backend implementations were checked against these versioned sources:

- [vLLM 0.26 custom logits processors](https://docs.vllm.ai/en/v0.26.0/features/custom_logitsprocs/)
- [vLLM 0.26 sampling parameters](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/sampling_params.py)
- [vLLM 0.26 sampler ordering](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/v1/sample/sampler.py)
- [vLLM 0.26 chat protocol](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/entrypoints/openai/chat_completion/protocol.py)
- [Transformers 5.14.1 generation](https://raw.githubusercontent.com/huggingface/transformers/v5.14.1/src/transformers/generation/utils.py)
- [Transformers 5.14.1 logits processors](https://raw.githubusercontent.com/huggingface/transformers/v5.14.1/src/transformers/generation/logits_process.py)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

The vLLM 0.26 chat protocol exposes `logprob_token_ids`, although some SamplingParams prose
describes it as engine-only. Endpoint tests are needed to establish what a deployment supports.
