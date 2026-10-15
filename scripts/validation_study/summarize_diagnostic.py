"""Reduce retained adapter/runtime traces without assuming absolute-logit parity."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from .common import read_json, write_json


def log_odds_error(first: dict[str, float], second: dict[str, float]) -> float:
    if set(first) != set(second) or len(first) < 2:
        raise ValueError("comparison requires identical action sets with at least two actions")
    return max(
        abs((first[a] - first[b]) - (second[a] - second[b])) for a, b in combinations(first, 2)
    )


def summarize(root: Path) -> dict:
    records = {
        mode: read_json(root / f"diagnostic-{mode}.json")
        for mode in ("transformers", "vllm-native")
    }
    result = {"scope": "16 fixed synthetic prompts; maxima over both action sets", "backends": {}}
    for mode, record in records.items():
        rows = record["rows"]
        expected = {(case, i) for case in ("binary", "multiclass") for i in range(8)}
        if {(r["case"], r["context_id"]) for r in rows} != expected or len(rows) != 16:
            raise ValueError("incomplete or duplicated diagnostic contexts")
        keys = (
            [
                "repeat",
                "direct",
                "last_token_only",
                "no_cache",
                "eager_attention",
                "fp32_head",
                "fp32_head_rounded_bf16",
            ]
            if mode == "transformers"
            else ["repeat", "direct", "batch"]
        )
        result["backends"][mode] = {
            "contexts": len(rows),
            "api_prompt_ids_equal": all(r["api_prompt_ids_equal"] for r in rows),
            "api_action_ids_equal": all(r["api_action_ids_equal"] for r in rows),
            "vs_adapter": {
                key: {
                    "max_absolute_score_error": max(
                        abs(r[key][a] - r["adapter"][a]) for r in rows for a in r["adapter"]
                    ),
                    "max_log_odds_error": max(log_odds_error(r[key], r["adapter"]) for r in rows),
                }
                for key in keys
            },
        }
        if mode == "vllm-native":
            result["backends"][mode]["runtime_prompt_ids_equal"] = all(
                r["runtime_prompt_ids_equal"] for r in rows
            )
    tf_rows = {(r["case"], r["context_id"]): r for r in records["transformers"]["rows"]}
    native_rows = records["vllm-native"]["rows"]
    result["cross_runtime"] = {
        key: max(
            log_odds_error(tf_rows[(r["case"], r["context_id"])][key], r["direct"])
            for r in native_rows
        )
        for key in ["adapter", "eager_attention", "fp32_head"]
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.root / "diagnostic-summary.json", summarize(args.root))


if __name__ == "__main__":
    main()
