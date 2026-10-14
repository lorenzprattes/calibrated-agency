# Changes

## 0.2.0a1

- Canonicalize API scores and returned maps to requested action order before sampling,
  so the ignored server sample and alternative order cannot change seeded client decisions.
- Add continuing caller-owned Python RNGs to core/API decision methods. Reject ambiguous
  seed/RNG combinations; native engine sampling retains its request-seed semantics.
- Expose `bayesian_correct_log` for finite logged priors/targets and public synchronous
  and asynchronous scorer protocols for host-provided transports.
- Validate score/probability consistency, decision draw bounds and selected probability,
  and matching action sets in scored decisions. Add focused public-contract regressions.
- Align candidate token discovery across all backends with GABM and DeSiRe: one bare-label
  token, exact decoding, distinct IDs. Remove the combined-prefix re-tokenization requirement.
- Require API tokenization/generation prompt parity for scoring and continuation, disable
  thinking at this action boundary, neutralize score-request penalties, and reject conflicting
  duplicate or malformed candidate scores.
- Permit floating-point zero probabilities in sampling and averaging. Normalize positive
  prior/target weights in log space and large probability weights without sum overflow.
- Fit marginal intercepts using log-domain updates. Report zero unsmoothed prior mass and
  unrepresentable score ranges explicitly.
- Require positive native decision temperatures; greedy choice cannot realize a target sampler.
- Compare backends by action identity independently of map order; validate parity tolerances
  and nonempty beta comparisons. Fingerprint immutable result maps without deepcopy errors.
- Include token-resolution semantics in prior cache identity, now schema version 3. Previous
  fingerprints intentionally differ. This helper is not a DeSiRe cache-file implementation.
- Add CPU source-level GABM/DeSiRe parity checks and verify isolated wheel installation.

This is a local alpha build. Publication and fresh live-backend acceptance remain separate.
