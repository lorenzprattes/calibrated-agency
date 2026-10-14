"""Canonical cache and provenance fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize({field.name: getattr(value, field.name) for field in fields(value)})
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("fingerprint mapping keys must be strings")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple | list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set | frozenset):
        canonical_items = [_canonicalize(item) for item in value]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False),
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprint floats must be finite")
        return value
    if value is None or isinstance(value, str | int | bool):
        return value
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize supported values deterministically and without insignificant whitespace."""
    return json.dumps(
        _canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prior_cache_identity(
    *,
    backend: str,
    backend_version: str,
    model: str,
    model_revision: str,
    tokenizer_revision: str,
    chat_template_hash: str,
    prompt_hash: str,
    action_schema: Sequence[Mapping[str, str]],
    token_ids: Mapping[str, int],
    context_dataset_hash: str,
    estimator: str,
    estimator_settings: Mapping[str, Any],
    sample_count: int,
    score_mode: str,
    generation_settings: Mapping[str, Any],
    token_resolution: str = "bare_label_exact_roundtrip_v1",
) -> dict[str, Any]:
    """Build the minimum identity used to key a reusable prior estimate."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return {
        "schema_version": 3,
        "token_resolution": token_resolution,
        "backend": backend,
        "backend_version": backend_version,
        "model": model,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "chat_template_hash": chat_template_hash,
        "prompt_hash": prompt_hash,
        "action_schema": list(action_schema),
        "token_ids": dict(token_ids),
        "context_dataset_hash": context_dataset_hash,
        "estimator": estimator,
        "estimator_settings": dict(estimator_settings),
        "sample_count": sample_count,
        "score_mode": score_mode,
        "generation_settings": dict(generation_settings),
    }
