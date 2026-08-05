# Contributing

## Development setup

```bash
uv sync --group dev
uv run pytest --cov
uv run ruff check .
uv run mypy
uv build
```

Keep the dependency-free mathematical core separate from optional serving runtimes. A new
backend must fail closed when score semantics or token boundaries are unknown.

## Changes to the method

A change to correction order, normalization, prior estimation, temperature handling, or
candidate scoring is a methodological change, not a refactor. Include:

- the mathematical definition;
- a regression/property test;
- expected effects at beta 0 and 1;
- a live backend parity test when applicable;
- documentation of any changed guarantee.

Do not weaken validation or insert silent fallbacks to make an unsupported endpoint appear
compatible.

## Data and examples

Do not add scraped, personal, licensed, or model-generated data without documenting its
source and redistribution terms. Synthetic fixtures are preferred for package tests.

## Pull requests

Keep changes focused. Report the commands run and their results. GPU results must include the
model revision, runtime versions, hardware, seeds, and retained machine-readable output.
