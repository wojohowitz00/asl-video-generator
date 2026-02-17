# Codebase Concerns

Analysis Date: 2026-02-17

## Critical Functional Gaps

- `vocabulary_generator.py` still uses placeholder behavior for wSignGen inference and model loading.
- `avatar_renderer.py` mesh rendering path is placeholder and does not assemble final mesh video output.

## Reliability Gaps

- No CI workflow enforcing pytest + lint + type checks.
- End-to-end coverage for vocabulary->mesh->video path is missing.

## Documentation Drift

- README examples include command names that do not match current script entry points.

## Operational Notes

- Beads daemon socket binding fails in this environment, causing direct-mode fallback (tracking still works).
