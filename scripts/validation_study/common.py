from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

CASES = {
    "binary": {"post": "post", "comment": "comment"},
    "multiclass": {
        "post": "post",
        "reply": "reply",
        "react": "react",
        "unreact": "undo",
        "reshare": "share",
        "follow": "follow",
        "unfollow": "unsubscribe",
        "join_community": "join",
        "leave_community": "leave",
        "do_nothing": "wait",
        "stop": "stop",
    },
}
MODES = ("api", "vllm-native", "transformers")
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def health_failure(
    *,
    phase: str,
    elapsed: float,
    progress_age: float,
    idle_for: float,
    startup_limit: float = 600,
    active_limit: float = 180,
    idle_limit: float = 120,
) -> str | None:
    if phase in {"starting", "server_start", "model_load"}:
        return "startup_timeout" if elapsed > startup_limit else None
    if progress_age > active_limit:
        return "no_progress"
    if progress_age > idle_limit and idle_for > idle_limit:
        return "idle_without_progress"
    return None


def now() -> float:
    return time.time()
