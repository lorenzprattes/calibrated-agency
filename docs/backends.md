# Model backends

Every scorer returns all candidate scores at the same next-token position, their token IDs,
and probabilities normalized over the candidate set. These are probabilities conditional
on choosing one of the listed actions, not full-vocabulary probabilities.

Labels are encoded on their own, without special tokens. Each must produce one distinct
token ID and decode exactly to the label. The prompt prefix is rendered separately.
Encoding the prefix and label together can merge tokens, but generation cannot change
prefix tokens it has already emitted.

`ActionScorer` defines synchronous `score(request)` and `AsyncActionScorer` defines its
asynchronous equivalent. Both take `ScoreRequest` and return `ScoreResult`. The async
protocol is a typing interface, not an HTTP client. Applications supply context fitting,
timeouts, cancellation, retries, and caching.

## vLLM API

The HTTP adapter is written for vLLM `0.26.*`. Start the server with
`--logprobs-mode raw_logprobs` or `raw_logits` and `--generation-config vllm`.
The adapter needs `/tokenize`, `/detokenize`, assistant-message continuation, and generation
responses containing prompt token IDs and counts. It supports at most 128 action tokens.
Thinking is disabled at the API action boundary. Speculative decoding is unsupported.

A score request includes:

```json
{
  "max_tokens": 1,
  "temperature": 1.0,
  "top_p": 1.0,
  "allowed_token_ids": [101, 102],
  "logprobs": true,
  "logprob_token_ids": [101, 102],
  "return_tokens_as_token_ids": true,
  "return_token_ids": true
}
```

The adapter also disables top-k, min-p, repetition, frequency, and presence penalties for
scoring. The caller must declare the server's raw score mode: the adapter records it but
cannot select or verify it through the request.

Before scoring or continuation, the adapter tokenizes the exact message and template settings
with `add_special_tokens=false`. Returned `prompt_token_ids` and `usage.prompt_tokens`
must match this result. Missing or changed metadata raises an error. The adapter does not
check context limits or shrink prompts.

`score()` records the server's uncorrected sample in `ScoreResult.selected_action`.
`decide()` uses the returned scores to sample a corrected action in Python. All result maps
follow the requested action order, so the server sample and response ordering cannot change
the client choice for a fixed seed.

Use `decide(..., rng=task_rng)` for a continuing client random sequence. Supplying both
`rng` and `seed` raises an error before making a request. If neither is supplied, the
client uses the request seed. Native engine sampling has its own seed semantics.

Chat continuation resends and re-renders the history. Check assistant prefill and parsed
continuations on the model you intend to use. For token-ID continuation, use a transport
that sends integer prompt IDs through the completions endpoint.

## vLLM native

Register the processor when constructing the model:

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

`make_vllm_extra_args` builds the per-request configuration for
`SamplingParams.extra_args`. Choose one activation mode:

- `apply_at_first_token=True` for a prompt already ending at the action boundary;
- `trigger_token_ids=[...]` for generation that reaches an action tag during the response.

The processor corrects action logits and masks other vocabulary entries at that position
only. It can change the highest-scoring token and is incompatible with vLLM 0.26 speculative
decoding.

Temperature compensation accounts for the runtime's later division by temperature.
Disable it only when deliberately reproducing uncorrected temperature handling.
`decide_native()` requires a finite positive temperature. The lower-level processor can
transform logits for greedy decoding, but those choices do not sample its softmax.

## Transformers native

`TransformersNativeBackend.score` renders the chat template and obtains raw next-token logits
from a direct forward pass.

`FirstTokenBayesianLogitsProcessor` corrects the first generated position after a known
prompt length. `decide_native` applies it to one forward pass and samples with a local
`torch.Generator`, bypassing checkpoint generation penalties and truncation filters.

If you use the processor with `model.generate`, configure generation explicitly. Inherited
penalties, biases, forced or suppressed tokens, and truncation filters can alter the result.
Beams, per-row action sets, heterogeneous padded prompt lengths, and speculative decoding
are unsupported.

## Multi-token labels and versions

A one-token score cannot represent a multi-token action. Supporting one would require
scoring each candidate sequence, with a rule for shared prefixes and sequence length.
The current adapters reject such labels.

The vLLM extra pins `0.26.*` because its processor interface changes between versions.
The Transformers extra uses the 5.x range selected for vLLM 0.26 compatibility. Model,
tokenizer, chat-template, PyTorch, and hardware versions still matter when reproducing results.
Fresh tests of the changed 0.2.0a1 adapters are pending; the retained GPU results cover 0.1.0.
