# Codebase Concerns

Analysis Date: 2026-02-17 (updated after v5u completion)

## Critical Functional Gaps

- Software 3D mesh backend and artifact-level fidelity regression tests are now in place.
- Optional `pyrender` backend path remains dependency-sensitive and should stay guarded by fallback behavior tests.

## Reliability Gaps

- CI lint gate now covers `src/`, `scripts/`, and `tests/`.
- CI mypy gate now covers four core modules (`vocabulary_generator`, `config`, `lesson_parser`, `pose_dictionary`).
- Additional strict typing debt remains in other modules and can be expanded incrementally in future phases.

## Documentation Drift

- Primary README command drift was corrected in v0.2; documentation must stay aligned as v0.3 introduces additional rendering controls.

## Operational Notes

- Beads is pinned to direct SQLite mode in `.beads/config.yaml` to avoid socket-binding failures.
- Issue DB and JSONL export were re-synchronized after backend cleanup (`bd export -o .beads/issues.jsonl`).
