# Backend contracts

## Common contract

A scorer must return, at one exact autoregressive boundary:

- every candidate's token ID;
- every candidate's raw model score;
- the probabilities obtained by renormalizing those scores over the declared candidate set;
- model/backend provenance.

The normalized candidate probabilities are conditional on choosing from the candidate set.
They are not unconditional full-vocabulary probabilities.

Labels must be distinct single tokens at the rendered boundary. Encoding each label in
isolation is not sufficient: byte-pair tokenization can merge across the prefix boundary.

## vLLM API

Supported deployment profile:

- vLLM `0.26.*`;
- server started with `--logprobs-mode raw_logprobs` (recommended) or `raw_logits`,
  plus `--generation-config vllm` to avoid inherited model-card sampling filters;
- no speculative decoding;
- no penalties, top-k, top-p, or logit bias in the score request;
- at most 128 candidate token IDs.

The adapter uses:

```json
{
  "max_tokens": 1,
  "temperature": 1.0,
  "allowed_token_ids": [101, 102],
  "logprobs": true,
  "logprob_token_ids": [101, 102],
  "return_tokens_as_token_ids": true
}
```

The score mode is configured server-wide and cannot be selected per request. The backend
therefore requires an explicit caller attestation and records it in every result. A public
endpoint that does not disclose its mode is not a supported deployment.

`score()` returns the server's constrained but uncorrected sample in
`ScoreResult.selected_action` for diagnostics. `decide()` ignores that sample, applies the
formula to the returned raw scores, and samples with its own seeded RNG; it returns both the
score response and corrected decision.

Chat continuation resends and re-renders the complete history; vLLM does not provide a
persistent conversation handle. For strict token-level continuation, prefer the completions
endpoint with integer prompt IDs. Chat prefill using `continue_final_message` needs a live
model-specific conformance test.

## vLLM native

Register the processor when constructing `LLM`:

```python
from vllm import LLM

llm = LLM(
    model=model_id,
    logits_processors=[
        "calibrated_agency.backends.vllm_native:BayesianActionLogitsProcessor"
    ],
    logprobs_mode="raw_logprobs",
)
```

Per-request configuration is passed through `SamplingParams.extra_args`, built by
`make_vllm_extra_args`.

Two activation modes are available:

- `apply_at_first_token=True`, for a prompt prefilled to the action boundary;
- `trigger_token_ids=[...]`, for paper-style one-pass generation that reaches an action tag.

Exactly one mode is required. The processor masks all other vocabulary entries only at the
activation step. It is not argmax-invariant and is incompatible with vLLM 0.26 speculative
decoding.

The processor compensates for vLLM's later temperature division. Disable compensation only
when intentionally reproducing an implementation that did not account for processor order.

## Transformers native

`TransformersNativeBackend.score` renders the chat template and executes a direct forward
pass. This is the simplest way to obtain raw next-token logits.

`FirstTokenBayesianLogitsProcessor` provides a native `model.generate` path. It records the
initial prompt width, applies the correction only at that first generated position, masks
other tokens, and compensates for the later Transformers temperature warper.

The first release does not support beams, per-row action sets, padded heterogeneous prompt
lengths, or speculative decoding. Fail rather than silently using these unsupported modes.

## Multi-token labels

A one-token score does not represent a multi-token choice. Multi-token actions require
teacher-forced sequence scoring along every candidate path, including careful treatment of
shared prefixes and length normalization. That is a separate future backend contract; the
current fast path rejects it.

## Version policy

vLLM's V1 custom-processor interface is evolving, so the optional extra pins `0.26.*`.
Transformers is constrained to the tested 5.x range compatible with vLLM 0.26. Hardware,
PyTorch, model, tokenizer, and chat-template revisions remain part of reproducibility
metadata.
