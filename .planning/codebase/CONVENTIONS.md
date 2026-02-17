# Conventions

Analysis Date: 2026-02-17

## Language and Style

- Python 3.11 project with type hints.
- Linting configured via Ruff (`E,F,I,UP,B`).
- Type checking configured via mypy (`strict = true`, with missing imports ignored).

## Testing

- pytest-based tests under `tests/`.
- Current verified baseline: 29 passing tests.

## Packaging and Commands

- Managed with `uv` and `pyproject.toml` script entry points.
- Prefer invoking via `uv run <script>`.
