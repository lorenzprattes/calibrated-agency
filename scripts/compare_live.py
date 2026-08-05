#!/usr/bin/env python3
"""Compare retained live-backend artifacts and enforce predeclared tolerances."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from calibrated_agency import bayesian_correct

THRESHOLDS = {
    "api_vs_vllm_native": {"log_odds": 0.001, "probability": 0.0001},
    "api_vs_transformers": {"log_odds": 0.05, "probability": 0.01},
}
NATIVE_PROCESSOR_TOLERANCE = 1e-5
BETA_ZERO_TOLERANCE = 1e-7


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def score_by_context(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["context_id"]): item for item in payload["scores"]}


def compare_pair(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    reference_scores = score_by_context(reference)
    candidate_scores = score_by_context(candidate)
    if set(reference_scores) != set(candidate_scores):
        raise ValueError(f"{name} context IDs do not match")
    prior = reference["probability_mean_prior"]
    target = reference["target"]
    log_odds_errors: list[float] = []
    probability_errors: list[float] = []
    token_mismatches = 0
    for context_id in reference_scores:
        first = reference_scores[context_id]
        second = candidate_scores[context_id]
        actions = tuple(first["raw_scores"])
        if len(actions) != 2 or set(actions) != set(second["raw_scores"]):
            raise ValueError("live comparison currently expects matching binary actions")
        first_log_odds = first["raw_scores"][actions[0]] - first["raw_scores"][actions[1]]
        second_log_odds = second["raw_scores"][actions[0]] - second["raw_scores"][actions[1]]
        log_odds_errors.append(abs(first_log_odds - second_log_odds))
        if first["token_ids"] != second["token_ids"]:
            token_mismatches += 1
        for beta in (0.0, 0.15, 0.5, 1.0):
            first_corrected = bayesian_correct(first["raw_scores"], prior, target, beta)
            second_corrected = bayesian_correct(second["raw_scores"], prior, target, beta)
            probability_errors.extend(
                abs(
                    first_corrected.probabilities[action]
                    - second_corrected.probabilities[action]
                )
                for action in actions
            )
    threshold = THRESHOLDS[name]
    result = {
        "name": name,
        "contexts": len(reference_scores),
        "max_absolute_log_odds_error": max(log_odds_errors),
        "mean_absolute_log_odds_error": math.fsum(log_odds_errors) / len(log_odds_errors),
        "max_corrected_probability_error": max(probability_errors),
        "token_id_mismatches": token_mismatches,
        "thresholds": threshold,
    }
    result["passed"] = (
        result["max_absolute_log_odds_error"] <= threshold["log_odds"]
        and result["max_corrected_probability_error"] <= threshold["probability"]
        and token_mismatches == 0
    )
    return result


def processor_check(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[float] = []
    beta_zero_errors: list[float] = []
    missing_reported = 0
    target = payload["target"]
    for item in payload["native_decisions"]:
        reported = item["backend_reported_probabilities"]
        if reported is None:
            missing_reported += 1
            continue
        for action, expected in item["expected_probabilities"].items():
            errors.append(abs(expected - reported[action]))
        if item["beta"] == 0.0:
            beta_zero_errors.extend(abs(reported[action] - target[action]) for action in target)
    maximum = max(errors, default=math.inf)
    beta_zero_maximum = max(beta_zero_errors, default=math.inf)
    return {
        "backend": payload["mode"],
        "decisions": len(payload["native_decisions"]),
        "missing_backend_reported": missing_reported,
        "max_expected_vs_backend_probability_error": maximum,
        "beta_zero_max_target_error": beta_zero_maximum,
        "tolerance": NATIVE_PROCESSOR_TOLERANCE,
        "passed": (
            missing_reported == 0
            and maximum <= NATIVE_PROCESSOR_TOLERANCE
            and beta_zero_maximum <= BETA_ZERO_TOLERANCE
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live backend parity",
        "",
        f"Model: `{report['model']}`",
        "",
        "## Score parity",
        "",
        "| Comparison | Max log-odds error | Max corrected-probability error "
        "| Token mismatches | Pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    for item in report["score_parity"]:
        lines.append(
            f"| {item['name']} | {item['max_absolute_log_odds_error']:.6g} | "
            f"{item['max_corrected_probability_error']:.6g} | "
            f"{item['token_id_mismatches']} | {'yes' if item['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Native processor parity",
            "",
            "| Backend | Decisions | Max expected/reported error | β=0 target error | Pass |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for item in report["native_processor_parity"]:
        lines.append(
            f"| {item['backend']} | {item['decisions']} | "
            f"{item['max_expected_vs_backend_probability_error']:.6g} | "
            f"{item['beta_zero_max_target_error']:.6g} | "
            f"{'yes' if item['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Required implementation checks pass: **{report['passed']}**",
            "",
            "Cross-runtime vLLM/Transformers diagnostic passes its strict tolerance: "
            f"**{report['cross_runtime_diagnostic_passed']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--vllm-native", type=Path, required=True)
    parser.add_argument("--transformers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    api = load(args.api)
    vllm_native = load(args.vllm_native)
    transformers = load(args.transformers)
    models = {api["model"], vllm_native["model"], transformers["model"]}
    if len(models) != 1:
        raise ValueError(f"backend model IDs differ: {models}")
    report = {
        "model": api["model"],
        "score_parity": [
            compare_pair(api, vllm_native, name="api_vs_vllm_native"),
            compare_pair(api, transformers, name="api_vs_transformers"),
        ],
        "native_processor_parity": [
            processor_check(vllm_native),
            processor_check(transformers),
        ],
    }
    report["cross_runtime_diagnostic_passed"] = report["score_parity"][1]["passed"]
    report["passed"] = report["score_parity"][0]["passed"] and all(
        item["passed"] for item in report["native_processor_parity"]
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.report.write_text(markdown(report), encoding="utf-8")
    print(args.report)
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
