# Testing

Analysis Date: 2026-02-17

## Test Framework

- pytest with tests in `tests/`

## Current Status

- Verified locally: `29 passed` via `uv run --extra dev python -m pytest -q`

## Covered Areas

- Config behavior (`test_config.py`)
- Gloss translation (`test_gloss_translator.py`)
- Pose dictionary (`test_pose_dictionary.py`)
- Pose generation (`test_pose_generator.py`)

## Known Gaps

- No end-to-end vocabulary->mesh->video coverage.
- No CI-enforced quality gates yet.
