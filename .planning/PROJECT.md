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
- Unit/integration tests currently passing locally (`29 passed`).
- Beads tracking initialized and backlog seeded for next milestone.
- Significant mesh/model paths remain placeholder implementations.

## Next Milestone Goals

- Replace placeholder mesh/model logic with production-ready implementations.
- Add end-to-end coverage for vocabulary -> mesh -> video flow.
- Add CI quality gates so regressions are caught automatically.
- Align documentation with current CLI entry points.

## Requirements

See `.planning/REQUIREMENTS.md`.

## Context

- Python 3.11 package managed by `uv`.
- CLI entry points in `pyproject.toml`: `asl-generate`, `asl-translate`, `asl-pose`, `asl-render`.
- Core modules are under `src/asl_video_generator/`.
- Content pipeline scripts are under `scripts/`.
- Backlog tracked in Beads epic `asl-video-generator-cxv`.

## Constraints

- Preserve current working tests while implementing missing functionality.
- Keep runtime-compatible workflows for local and cloud environments.
- Prefer deterministic artifacts and explicit manifests for downstream app use.
- Keep docs/CLI examples accurate to current entry points.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Track work in Beads | Persistent, dependency-aware backlog | In use (`asl-video-generator-cxv`) |
| Use GSD `.planning` docs | Make phase status explicit and resumable | Initialized |
| Prioritize real mesh/model implementation next | Placeholder logic blocks production usefulness | Planned in Phase 3 |
| Gate E2E tests behind feature completion | Prevent flaky or misleading coverage | Dependency-linked in Beads |

---
*Last updated: 2026-02-17*
