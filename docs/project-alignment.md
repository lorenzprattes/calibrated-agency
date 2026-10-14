# Using calibrated-agency with DeSiRe

The package and DeSiRe implement the same Bayesian correction. DeSiRe still uses its own
math module and asynchronous transport. The package has not yet replaced them.

## Math and action selection

The original Reddit GABM implementation uses two actions, `post` and `comment`.
The package accepts an ordered action set supplied by the application. DeSiRe uses a larger
menu that changes with the actions and targets currently available to an agent.

Both support the paper's closed form. Probability-mean priors and fitted marginal intercepts
are extensions to the original approach. Fitted intercepts match the target on the supplied
calibration sample; the closed form generally does not at positive beta.

The interfaces needed by DeSiRe are available:

- `bayesian_correct_log` accepts its finite logged priors and targets directly.
- `decide(..., rng=task_rng)` and `sample_probabilities(..., rng=task_rng)` continue the
  session's random sequence rather than restarting it on every call.
- `ActionSpace` separates internal action names from model labels.
- `AsyncActionScorer` defines `async def score(request: ScoreRequest) -> ScoreResult`
  for a transport supplied by the application.

The API adapter preserves the declared action order when parsing scores. Changing the
server's sampled token or the order of its score alternatives does not change the seeded
client decision. Result objects check that probabilities match scores and that action sets,
selected probabilities, and random draws are valid.

## What stays in DeSiRe

DeSiRe constructs and fits model context, derives legal actions, parses continuations,
handles retries and timeouts, executes platform actions, and records trajectories.
It also manages the calibration score cache and checks deployment provenance.
Those responsibilities remain in the harness.

The package's HTTP adapter is synchronous. DeSiRe can keep its existing asynchronous
transport and implement `AsyncActionScorer`; a second HTTP client is unnecessary.
The protocol itself adds no retry loop or cache.

Use the runtime's action-label map when creating an `ActionSpace`. In the consolidated
DeSiRe version, `undo` represents `unreact`, `unsubscribe` represents `unfollow`, and
`share`, `join`, `leave`, and `wait` also map to different internal names. Encoding an
internal enum name directly may therefore select the wrong model token.

Both implementations resolve bare labels without special tokens and check their exact
decoding. They render the scoring prefix independently: generation cannot replace tokens
that have already been emitted by re-tokenizing the prefix and label together.

## Checks completed

The original GABM processor at `def1eaa6508f21bbaf0ac5fee2147c6dcd201d25` was compared with
the initial 0.2.0a1 package snapshot and two DeSiRe revisions:

| DeSiRe revision | Checkout | Focused harness tests |
|---|---|---:|
| `92339eb35639e49fbc9ad511f45b2c8bd3fa55ab` | Root, with unrelated working changes | 114 passed |
| `be233f18b29256d5f382bb307442775bbabdc730` | Consolidated architecture | 150 passed |

The source-level comparison checked 160 binary corrections, 480 additional multiclass
corrections, prior estimation, fitted distributions, and 12,800 seeded choices per checkout.
The maximum closed-form probability difference was below `5e-16`, with no sampled-action
mismatches. These checks executed selected source functions with simulated tensor access;
they did not run a model or the GABM simulation.

The later public-API and sampling fixes have focused package regression tests. Cross-project
checks have not been repeated after those changes. The original reports retain the source
hashes of the snapshot they tested. See [CPU results](../validation/cpu-2026-10-14/README.md).

For a future comparison, the existing source checker accepts checkout paths:

```bash
uv run python scripts/check_project_alignment.py \
  --gabm /path/to/bayesian-inference-gabm \
  --digital-twin /path/to/digital-twin
```

## Remaining work

Actual harness replacement and its integration checks remain pending. No separate replacement
acceptance suite has been added.

The current HTTP adapter does not verify model/tokenizer revisions, server version, context
limits, or launch flags automatically. The caller declares the score mode. DeSiRe already
performs additional checks and context fitting, which should remain in place during migration.

Calibration depends on the contexts and scheduling profile used to obtain scores. DeSiRe's
feed-subset sampling may not represent later simulation contexts, and model scores can change
with batching. Matching the correction formula does not resolve either issue.

Fresh live tests are still needed for the changed adapters and every action label used by a
deployment. Multi-token labels, speculative decoding, and unobserved reasoning before the API
action boundary remain unsupported. See [backend setup](backends.md) and
[validation](validation.md) for the test conditions.
