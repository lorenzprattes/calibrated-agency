"""Execute source-level parity checks against local GABM and DeSiRe checkouts.

Only the selected pure functions/processor method are compiled from the GABM
source. Tensor access is simulated; this does not execute vLLM or a model.
The DeSiRe math module is loaded unchanged. No source checkout is modified.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from calibrated_agency import (
    __version__,
    apply_intercepts,
    bayesian_correct,
    counted_prior,
    fit_marginal_intercepts,
    probability_mean_prior,
    sample_probabilities,
    softmax,
)
from calibrated_agency.fingerprint import file_sha256


def source_function(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            selected,
        ],
        type_ignores=[],
    )
    namespace: dict[str, Any] = {"math": math, "ACTION_TOKENS": ["post", "comment"]}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class Tensor:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __getitem__(self, index: tuple[int, int]) -> Scalar:
        return Scalar(self.values[index[1]])

    def __setitem__(self, index: tuple[int, int | slice], value: float) -> None:
        if isinstance(index[1], slice):
            self.values[:] = [value] * len(self.values)
        else:
            self.values[index[1]] = value


def revision(root: Path) -> dict[str, Any]:
    return {
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
            ).strip()
        ),
    }


def check_alignment(gabm: Path, twin: Path) -> dict[str, Any]:
    processor_path = gabm / "subreddit_simulation/logits_processor.py"
    prior_path = gabm / "subreddit_simulation/prior_estimation.py"
    twin_path = twin / "harness/src/harness/bayesian_math.py"
    reference_apply = source_function(processor_path, "apply")
    reference_counts = source_function(prior_path, "_counts_to_priors")
    spec = importlib.util.spec_from_file_location("alignment_twin_math", twin_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load digital twin math")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rng = random.Random(20261014)
    max_reference_error = 0.0
    max_twin_error = 0.0
    max_prior_error = 0.0
    max_fit_error = 0.0
    max_sample_error = 0.0
    binary_cases = 0
    multiclass_cases = 0
    for count in (2, 3, 5, 11):
        actions = ("post", "comment") if count == 2 else tuple(f"action_{i}" for i in range(count))
        for _ in range(40):
            raw = {action: rng.uniform(-15, 15) for action in actions}
            prior = softmax({action: rng.uniform(-5, 5) for action in actions})
            target = softmax({action: rng.uniform(-5, 5) for action in actions})
            for beta in (0.0, 0.15, 0.5, 1.0):
                corrected = bayesian_correct(raw, prior, target, beta).probabilities
                twin_corrected = module.corrected_distribution(
                    raw,
                    {action: math.log(value) for action, value in prior.items()},
                    {action: math.log(value) for action, value in target.items()},
                    beta,
                )
                max_twin_error = max(
                    max_twin_error,
                    max(abs(corrected[action] - twin_corrected[action]) for action in actions),
                )
                first_rng, second_rng = random.Random(42), random.Random(42)
                for _ in range(20):
                    selected, _ = sample_probabilities(corrected, rng=first_rng)
                    twin_selected = module.sample_distribution(twin_corrected, second_rng)
                    max_sample_error = max(max_sample_error, float(selected != twin_selected))
                if count == 2:
                    state = SimpleNamespace(
                        fired=False,
                        trigger_matched=lambda: True,
                        beta=beta,
                        action_token_ids={"post": 0, "comment": 1},
                        log_prior={action: math.log(value) for action, value in prior.items()},
                        log_pi={action: math.log(value) for action, value in target.items()},
                    )
                    tensor = Tensor([*raw.values(), 99])
                    reference_apply(SimpleNamespace(requests={0: state}), tensor)
                    reference = softmax(dict(zip(actions, tensor.values[:2], strict=True)))
                    if tensor.values[2] != -math.inf:
                        raise AssertionError("reference failed to mask non-action vocabulary")
                    max_reference_error = max(
                        max_reference_error,
                        max(abs(corrected[action] - reference[action]) for action in actions),
                    )
                    binary_cases += 1
                else:
                    multiclass_cases += 1

        contexts = [{action: rng.uniform(-5, 5) for action in actions} for _ in range(10)]
        mean_prior = probability_mean_prior(
            [softmax(scores) for scores in contexts], pseudocount=1e-12
        )
        twin_prior = module.probability_mean_log_prior(contexts)
        max_prior_error = max(
            max_prior_error,
            max(abs(mean_prior[action] - math.exp(twin_prior[action])) for action in actions),
        )
        target = {action: i + 1.0 for i, action in enumerate(actions)}
        fit = fit_marginal_intercepts(contexts, target, 0.5)
        twin_fit = module.fit_marginal_intercepts(contexts, target, 0.5)
        for context in contexts:
            first = apply_intercepts(context, fit.intercepts, 0.5)
            second = module.apply_intercepts(context, twin_fit.intercepts, 0.5)
            max_fit_error = max(max_fit_error, max(abs(first[a] - second[a]) for a in actions))

    for post_count in range(11):
        counts = {0: {"post": post_count, "comment": 10 - post_count}}
        reference = reference_counts([SimpleNamespace(username="fixture")], counts)["fixture"]
        standalone = counted_prior(
            ["post"] * post_count + ["comment"] * (10 - post_count), ["post", "comment"]
        )
        max_prior_error = max(
            max_prior_error,
            max(abs(standalone[action] - math.exp(reference[action])) for action in standalone),
        )

    errors = {
        "gabm_closed_form": max_reference_error,
        "twin_closed_form": max_twin_error,
        "prior_estimators": max_prior_error,
        "twin_fitted_distributions": max_fit_error,
        "seeded_action_mismatches": max_sample_error,
    }
    passed = all(value <= 1e-9 for value in errors.values())
    return {
        "passed": passed,
        "package_version": __version__,
        "python": sys.version,
        "seed": 20261014,
        "scope": "CPU source-level numerical parity; simulated tensor, no model execution",
        "binary_corrections": binary_cases,
        "multiclass_corrections": multiclass_cases,
        "counted_prior_cases": 11,
        "seeded_decisions": 20 * (binary_cases + multiclass_cases),
        "tolerance": 1e-9,
        "maximum_errors": errors,
        "sources": {
            "gabm": revision(gabm),
            "digital_twin": revision(twin),
            "sha256": {
                str(path): file_sha256(path) for path in (processor_path, prior_path, twin_path)
            },
            "package_sha256": {
                str(path.relative_to(Path(__file__).resolve().parents[1])): file_sha256(path)
                for path in sorted(
                    (Path(__file__).resolve().parents[1] / "src/calibrated_agency").rglob("*.py")
                )
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gabm", type=Path, required=True)
    parser.add_argument("--digital-twin", type=Path, required=True)
    args = parser.parse_args()
    result = check_alignment(args.gabm, args.digital_twin)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
