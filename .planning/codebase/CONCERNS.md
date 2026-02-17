# Codebase Concerns

Analysis Date: 2026-02-17 (updated after q7m completion)

## Critical Functional Gaps

- Software 3D mesh backend and artifact-level fidelity regression tests are now in place.
- Optional `pyrender` backend path now has explicit routing + tested fallback behavior.
- Native high-fidelity pyrender rendering implementation is still not enabled (current pyrender path delegates to deterministic software fallback).

## Reliability Gaps

- CI lint gate now covers `src/`, `scripts/`, and `tests/`.
- CI mypy gate now covers full `src/asl_video_generator` scope (with `--follow-imports=skip`).
- Further type hardening opportunity remains for stricter modes (for example reducing reliance on `Any` across external client boundaries).

## Documentation Drift

- Primary README command drift was corrected in v0.2; documentation must stay aligned as v0.3 introduces additional rendering controls.

## Operational Notes

- Beads is pinned to direct SQLite mode in `.beads/config.yaml` to avoid socket-binding failures.
- Issue DB and JSONL export were re-synchronized after backend cleanup (`bd export -o .beads/issues.jsonl`).
