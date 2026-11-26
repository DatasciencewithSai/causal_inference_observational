# CLAUDE.md

Python package `causal` (src layout): causal inference for observational data and quasi-experiments.
Randomized A/B testing lives in the sister repo `experimentation_ab_testing`, not here.

## Commands
- Setup: `uv venv .venv && uv pip install -p .venv -e ".[dev]"` (or `pip install -e ".[dev]"`)
- Test: `.venv/bin/pytest -q`
- Lint/format: `.venv/bin/ruff check . && .venv/bin/ruff format --check .` (fix with `ruff format .`)

## Conventions
- One module per method family in `src/causal/`; export public functions in `__init__.py`.
- Every estimator returns a frozen dataclass with the point estimate, SE, p-value and CI.
- Docstrings state the identifying assumptions (parallel trends, unconfoundedness, overlap, ...).
- Tests simulate data with a known true effect (fixed seed) and check the estimate recovers it,
  and that a naive estimator is biased where the method should fix that.
- Keep dependencies to numpy and scipy unless there's a strong reason; next roadmap item may justify more.

## Workflow
- Run tests and ruff before every commit; CI (`.github/workflows/ci.yml`) must stay green.
- Work on a feature branch; don't push directly to `main` unless the owner asks.
- Pick new work from `docs/ROADMAP.md` and tick items off there.
