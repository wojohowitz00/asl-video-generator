# ASL Video Generator

## What This Is

ASL Video Generator is a Python pipeline that converts English text into ASL-oriented video outputs using:

1. LLM-based gloss translation
2. Pose/motion generation
3. Video rendering and publishing helpers

It also supports curriculum ingestion and S3 manifest publishing for downstream learning apps.

## Core Value

Provide a reproducible path from text input to ASL-compatible video assets that can be published and consumed by client applications.

## Current State

- Baseline pipeline implemented in Python package + CLI scripts.
- Curriculum and content-manifest publishing workflow is present.
- Unit/integration tests currently passing locally (`58 passed`).
- v0.2 milestone backlog (`asl-video-generator-cxv`) is complete and closed.
- v0.3 milestone backlog (`asl-video-generator-v5u`) is complete and closed.
- v0.4 typing tranche (`asl-video-generator-t9r`) is complete and closed.
- v0.5 pyrender hardening tranche (`asl-video-generator-q7m`) is complete and closed.
- v0.6 native pyrender backend MVP (`asl-video-generator-r6p`) is complete and closed.
- v0.7 render script routing fix tranche (`asl-dy5`) is complete and closed.
- v0.8 stylized routing completion tranche (`asl-o3d`) is complete and closed.
- v0.9 render3d CI smoke coverage tranche (`asl-cwy`) is complete and closed.
- v1.0 offscreen render smoke execution tranche (`asl-dmv`) is complete and closed.
- v1.1 render_mesh end-to-end smoke tranche (`asl-1ib`) is complete and closed.
- v1.2 pyrender fallback telemetry tranche (`asl-6tl`) is complete and closed.
- v1.3 pyrender camera-angle regression tranche (`asl-95a`) is complete and closed.
- v1.4 mesh backend benchmark baseline tranche (`asl-atw`) is complete and closed.
- v1.5 benchmark baseline artifact tranche (`asl-2la`) is complete and closed.
- CI workflow now runs pytest + repo-wide Ruff (`src/scripts/tests`) + full mypy scope for `src/asl_video_generator`.
- Beads is configured for stable direct SQLite usage in this environment (`.beads/config.yaml`).

## Next Milestone Goals

- Evaluate visual/performance quality of native `pyrender` path against software renderer baseline.
- Define next feature backlog for renderer robustness and quality after v1.5 benchmark artifact trendability baseline.

## Requirements

See `.planning/REQUIREMENTS.md`.

## Context

- Python 3.11 package managed by `uv`.
- CLI entry points in `pyproject.toml`: `asl-generate`, `asl-translate`, `asl-pose`, `asl-render`.
- Core modules are under `src/asl_video_generator/`.
- Content pipeline scripts are under `scripts/`.
- Current Beads backlog is empty (all known work closed through v1.5).

## Constraints

- Preserve current working tests while implementing missing functionality.
- Keep runtime-compatible workflows for local and cloud environments.
- Prefer deterministic artifacts and explicit manifests for downstream app use.
- Keep docs/CLI examples accurate to current entry points.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Track work in Beads | Persistent, dependency-aware backlog | In use (`asl-video-generator-v5u`, `asl-video-generator-t9r`, `asl-video-generator-q7m`) |
| Use GSD `.planning` docs | Make phase status explicit and resumable | Initialized |
| Complete v0.2 before widening scope | Keep momentum and de-risk integration first | Completed (Phases 3/4) |
| Plan v0.3 around fidelity + quality debt | Highest remaining product and engineering leverage | Completed (Phases 5/6) |
| Run v0.4 typing hardening tranche | Enable broader type-safety gate with low risk behavior change | Completed (`t9r.1`-`t9r.4`) |
| Run v0.5 pyrender hardening tranche | Stabilize optional backend behavior and avoid runtime surprises | Completed (`q7m.1`-`q7m.3`) |
| Run v0.6 native pyrender MVP tranche | Replace pyrender fallback stub with native offscreen rendering while preserving deterministic fallback | Completed (`r6p.1`-`r6p.3`) |
| Run v0.7 render script routing fix tranche | Ensure mesh avatar-style batch renders invoke mesh renderer path instead of pose renderer | Completed (`dy5.1`-`dy5.3`) |
| Run v0.8 stylized routing completion tranche | Ensure stylized avatar-style batch renders also invoke mesh renderer path | Completed (`o3d.1`-`o3d.3`) |
| Run v0.9 render3d CI smoke tranche | Validate optional render3d dependency path continuously in CI with dedicated smoke coverage | Completed (`cwy.1`-`cwy.3`) |
| Run v1.0 offscreen smoke execution tranche | Add real offscreen pyrender execution smoke assertion with safe skip on missing GL context | Completed (`dmv.1`-`dmv.2`) |
| Run v1.1 render_mesh E2E smoke tranche | Add render_mesh end-to-end output-generation smoke coverage for pyrender backend path | Completed (`1ib.1`-`1ib.2`) |
| Run v1.2 fallback telemetry tranche | Add runtime telemetry for effective backend usage and pyrender fallback counts with regression assertions | Completed (`6tl.1`-`6tl.3`) |
| Run v1.3 camera-angle regression tranche | Add pyrender backend artifact regression for camera-angle sensitivity with runtime-safe skip behavior | Completed (`95a.1`-`95a.2`) |
| Run v1.4 benchmark baseline tranche | Add reproducible software_3d vs pyrender runtime benchmark script with JSON report output and test coverage | Completed (`atw.1`-`atw.3`) |
| Run v1.5 benchmark artifact tranche | Commit baseline benchmark artifact, validate schema via tests, and document rerun/compare workflow | Completed (`2la.1`-`2la.3`) |

---
*Last updated: 2026-02-18*
