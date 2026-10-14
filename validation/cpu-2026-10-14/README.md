# CPU alpha validation

Local validation of `calibrated-agency 0.2.0a1` on 14 October 2026.
These checks ran without a model endpoint or GPU runtime.

The table and source-parity JSON files below describe the initial alpha snapshot.
The subsequent public-API and sampling fixes passed **104 package tests** on Python
3.10.20, 3.12.3, and 3.14.6, with **93.44% core branch coverage** on Python 3.14,
Ruff lint and strict mypy passing. Those changes have focused package regressions;
the source-parity checker and harness integration checks were not rerun after those fixes.
The local wheel and source distribution were rebuilt with the public-API fixes.

| Check | Result |
|---|---|
| Package tests, Python 3.10.20 | 88 passed |
| Package tests, Python 3.12.3 | 88 passed |
| Package tests, Python 3.14.6 | 88 passed |
| Core branch coverage, Python 3.10 | 93.94%; required threshold 90% |
| Ruff lint | Passed |
| Strict mypy | Passed, 12 package source files |
| GABM/root DeSiRe source parity | Passed |
| GABM/consolidated DeSiRe source parity | Passed |
| Root harness focused tests | 114 passed |
| Consolidated harness focused tests | 150 passed |
| Wheel and source distribution | Built |
| Isolated wheel installation, Python 3.12 | Passed |
| Synthetic binary example | Passed |

The installed-wheel check used Python isolated mode (`-I`) in a fresh environment with only
the wheel installed using `--no-deps`. It checked the version and typed-package marker,
zero-probability sampling, fitting from underflowed initial probabilities, and importability
of optional-backend modules without their runtime extras. It did not run those backends.

Both source-parity reports include the package version, interpreter, seed, tolerance,
source hashes, and inspected Git revisions/dirty state:

- [Root checkout](gabm-desire-root.json)
- [Consolidated architecture](gabm-desire-consolidated.json)

Reproduce package checks from the repository root:

```bash
uv sync --group dev --python 3.10
uv run pytest --cov
uv run ruff check .
uv run mypy
uv build
```

Run the same pytest suite in separate environments for 3.12 and 3.14. Versions 3.11 and
3.13 were not executed in this local validation.
Focused DeSiRe validation used:

```bash
python -m pytest -q \
  harness/tests/test_bayesian_math.py harness/tests/test_bayesian.py \
  harness/tests/test_priors.py harness/tests/test_server.py harness/tests/test_session.py
```

The [alignment checker](../../scripts/check_project_alignment.py) compares source-level
numerical transformations using simulated tensor access. It does not execute the GABM
simulation or establish current deployment tokenization, GPU sampling, behavioral validity,
or identical trajectories. Fresh live checks for the changed adapters remain required;
the [historical GPU reports](../gsc-2026-08-05/README.md) cover version 0.1.0.
