# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-02-18)

Core value: reproducible text-to-ASL video generation and publishing pipeline.
Current focus: v1.5 benchmark artifact trendability completed; preparing next backlog increment.

## Current Position

Phase: Milestone complete (through Phase 18)
Plan status: Complete
Last activity: 2026-02-18 - Completed `asl-2la` (benchmark artifact trendability)

Progress: [##########] v0.2 and v0.3 complete

## Performance Snapshot

- Local test status: `58 passed`
- Beads summary: `0 open`, `0 in_progress`, `0 blocked`, `0 ready`, `64 closed`
- Current blocked item: none

## Pending Todos

- No active Beads todos in this repository.

## Blockers / Concerns

- Current mypy gate is clean across `src/asl_video_generator` with `--follow-imports=skip`; stricter typing modes can be a future enhancement.
- Native `pyrender` backend path is now implemented with deterministic fallback on missing deps and render errors.
- AvatarRenderer now tracks effective mesh backend usage and pyrender fallback counts for runtime observability.
- `scripts/render_videos.py` now routes mesh and stylized avatar-style runs to `render_mesh` correctly.
- CI now includes a dedicated `render3d_smoke` job that installs optional `render3d` extras and runs targeted smoke tests.
- Render3d smoke coverage now includes an offscreen execution assertion with safe skip for missing GL context.
- Render3d smoke suite now covers `AvatarRenderer.render_mesh(...)` end-to-end artifact generation under pyrender backend.
- Render3d smoke suite now includes a camera-angle regression for pyrender backend artifact differences.
- Local benchmark script now provides a reproducible runtime baseline for requested software_3d vs pyrender backends.
- Benchmark baseline artifact is now committed and schema-validated for trendability across future runs.
- Beads is now pinned to direct SQLite mode for this environment (`.beads/config.yaml`).

## Session Continuity

Last validation run: `uv run --extra dev python -m pytest -q` on 2026-02-18 (`58 passed`)
Resume by running: `bd ready`, then create/plan next milestone issues.
