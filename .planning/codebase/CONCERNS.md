# Codebase Concerns

Analysis Date: 2026-02-18 (updated after dmv completion)

## Critical Functional Gaps

- Software 3D mesh backend and artifact-level fidelity regression tests are now in place.
- Optional `pyrender` backend path now has explicit routing + tested fallback behavior.
- Native pyrender offscreen rendering is now enabled for `mesh_backend=pyrender` when optional deps are installed.
- `scripts/render_videos.py` now routes mesh and stylized avatar-style runs to `render_mesh` (fixed from previous unconditional pose route).
- CI now includes a dedicated `render3d_smoke` job to validate optional extras installation and dependency probe behavior.
- Render3d smoke tests now include real offscreen render execution assertions with safe skip when GL context is unavailable.

## Reliability Gaps

- CI lint gate now covers `src/`, `scripts/`, and `tests/`.
- CI mypy gate now covers full `src/asl_video_generator` scope (with `--follow-imports=skip`).
- Further type hardening opportunity remains for stricter modes (for example reducing reliance on `Any` across external client boundaries).

## Documentation Drift

- Primary README command drift was corrected in v0.2; docs now also include `uv sync --extra render3d` guidance for native pyrender path.

## Operational Notes

- Beads is pinned to direct SQLite mode in `.beads/config.yaml` to avoid socket-binding failures.
- Issue DB and JSONL export were re-synchronized after backend cleanup (`bd export -o .beads/issues.jsonl`).
