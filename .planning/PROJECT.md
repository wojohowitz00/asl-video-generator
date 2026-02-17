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
- Unit/integration tests currently passing locally (`40 passed`).
- v0.2 milestone backlog (`asl-video-generator-cxv`) is complete and closed.
- v0.3 milestone backlog (`asl-video-generator-v5u`) is complete and closed.
- v0.4 typing tranche (`asl-video-generator-t9r`) is complete and closed.
- CI workflow now runs pytest + repo-wide Ruff (`src/scripts/tests`) + full mypy scope for `src/asl_video_generator`.
- Beads is configured for stable direct SQLite usage in this environment (`.beads/config.yaml`).

## Next Milestone Goals

- Decide whether to integrate optional high-fidelity external mesh backend (`pyrender` path) in production.
- Define next feature backlog after v0.3 completion.

## Requirements

See `.planning/REQUIREMENTS.md`.

## Context

- Python 3.11 package managed by `uv`.
- CLI entry points in `pyproject.toml`: `asl-generate`, `asl-translate`, `asl-pose`, `asl-render`.
- Core modules are under `src/asl_video_generator/`.
- Content pipeline scripts are under `scripts/`.
- Current Beads backlog is empty (all known work closed).

## Constraints

- Preserve current working tests while implementing missing functionality.
- Keep runtime-compatible workflows for local and cloud environments.
- Prefer deterministic artifacts and explicit manifests for downstream app use.
- Keep docs/CLI examples accurate to current entry points.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Track work in Beads | Persistent, dependency-aware backlog | In use (`asl-video-generator-v5u`, `asl-video-generator-t9r`) |
| Use GSD `.planning` docs | Make phase status explicit and resumable | Initialized |
| Complete v0.2 before widening scope | Keep momentum and de-risk integration first | Completed (Phases 3/4) |
| Plan v0.3 around fidelity + quality debt | Highest remaining product and engineering leverage | Completed (Phases 5/6) |
| Run v0.4 typing hardening tranche | Enable broader type-safety gate with low risk behavior change | Completed (`t9r.1`-`t9r.4`) |

---
*Last updated: 2026-02-17*
