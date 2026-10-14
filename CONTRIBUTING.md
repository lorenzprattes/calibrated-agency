# Contributing

## Setup

```bash
uv sync --group dev
uv run pytest --cov
uv run ruff check .
uv run mypy
uv build
```

Keep the core independent of model runtimes. Backend dependencies belong in optional extras.
Errors in scores or tokenization should raise an exception rather than silently change the
sampling method.

## Changes to the correction

Changes to normalization, prior estimation, temperature handling, or candidate scoring can
change the method's behavior. Explain the formula and add a focused regression test, including
the effects at beta zero and one. Record changes to the guarantees in the method documentation.
Run a live comparison when a change depends on model-runtime behavior, or state that it remains
untested. Keep the existing thresholds when reporting results.

## Examples and data

Use synthetic fixtures where possible. Document the source and redistribution terms of any
external data added to the repository.

## Pull requests

Describe the problem, the change, and the checks you ran. For GPU results, record model and
runtime versions, hardware, seeds, and machine-readable output so another developer can
repeat the run.
