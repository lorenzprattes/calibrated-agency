"""Backend protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from calibrated_agency.schemas import ScoreRequest, ScoreResult


@runtime_checkable
class ActionScorer(Protocol):
    """Backend contract required by the correction layer."""

    def score(self, request: ScoreRequest) -> ScoreResult:
        """Return every candidate's score at the exact next-token boundary."""
        ...


class AsyncActionScorer(Protocol):
    """Host-supplied asynchronous scoring with the same request/result contract.

    Implementations propagate cancellation and leave request fitting, retries,
    caching, and inference resource ownership to the host application.
    """

    async def score(self, request: ScoreRequest) -> ScoreResult:
        """Return every candidate's next-token score in action-space order."""
        ...


@dataclass(frozen=True)
class ContinuationResult:
    text: str
    finish_reason: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_response: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.raw_response is not None:
            object.__setattr__(self, "raw_response", MappingProxyType(dict(self.raw_response)))
