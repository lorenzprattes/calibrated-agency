from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

from .common import CASES, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--base-url")
    args = parser.parse_args()
    plan = read_json(args.root / "plan.json")
    output = args.root / "results" / args.mode
    output.mkdir(parents=True, exist_ok=True)

    def progress(step: str, phase: str = "active") -> None:
        write_json(output / "progress.json", {"updated": time.time(), "phase": phase, "step": step})

    progress("import", "model_load")
    live = importlib.import_module("scripts.live_backend")
    progress("model_load", "model_load")
    backend = live.load_backend(
        args.mode,
        plan["model"],
        args.base_url,
        revision=plan["model_revision"],
        gpu_memory_utilization=plan["gpu_memory_utilization"],
    )
    for case, labels in CASES.items():
        progress(f"{case}:start")
        result = live.run(
            args.mode,
            plan["model"],
            args.base_url,
            backend=backend,
            action_labels=labels,
            progress=lambda step, case=case: progress(f"{case}:{step}"),
        )
        result["study"] = plan
        write_json(output / f"{case}.json", result)
    progress("finished", "complete")


if __name__ == "__main__":
    main()
