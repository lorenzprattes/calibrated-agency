"""Candidate identity at a fixed autoregressive prefix.

Generation appends token IDs; it cannot retokenize already emitted prefix tokens.
Resolve bare labels and verify their exact decoding, as in GABM and DeSiRe.
"""

from collections.abc import Callable

from calibrated_agency.schemas import ActionSpace


def resolve_action_tokens(
    action_space: ActionSpace,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
) -> dict[str, int]:
    token_ids: dict[str, int] = {}
    for candidate in action_space.candidates:
        ids = encode(candidate.label)
        if len(ids) != 1 or type(ids[0]) is not int or ids[0] < 0:
            raise RuntimeError(f"label {candidate.label!r} must encode to one nonnegative token ID")
        decoded = decode(ids)
        if decoded != candidate.label:
            raise RuntimeError(
                f"token {ids[0]} decodes to {decoded!r}, expected exact label {candidate.label!r}"
            )
        token_ids[candidate.action] = ids[0]
    if len(set(token_ids.values())) != len(token_ids):
        raise RuntimeError("candidate labels resolve to duplicate token IDs")
    return token_ids
