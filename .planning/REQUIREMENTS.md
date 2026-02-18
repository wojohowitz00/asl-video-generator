# Requirements

## Baseline Pipeline (Delivered)

- [x] **PIPE-01**: Translate English text to ASL gloss sequences.
- [x] **PIPE-02**: Generate pose sequences from gloss outputs.
- [x] **PIPE-03**: Render pose-based videos.
- [x] **PIPE-04**: Support batch generation workflows.
- [x] **PIPE-05**: Publish generated content manifests for client apps.

## Productionization (Active)

- [x] **PROD-01**: Replace placeholder wSignGen inference path with real model-backed generation.
- [x] **PROD-02**: Replace placeholder mesh rendering with real video rendering pipeline.
- [x] **PROD-03**: Add robust end-to-end tests for vocabulary -> mesh -> video.
- [x] **PROD-04**: Add CI quality gates for tests, linting, and type checks.
- [x] **PROD-05**: Align README and operator docs with current CLI entry points.

## Operational Quality

- [x] **QUAL-01**: Local pytest suite passes in development environment.
- [x] **QUAL-02**: Automated CI validates every change.
- [x] **QUAL-03**: Critical pipeline paths have integration coverage.

## v0.3 Rendering Fidelity + Code Health (Planned)

- [x] **FID-01**: Upgrade mesh rendering from stylized 2D proxy to true 3D mesh rasterization.
- [x] **FID-02**: Expose mesh rendering backend/style controls in CLI and batch scripts.
- [x] **FID-03**: Add artifact-level regression tests for high-fidelity mesh rendering outputs.
- [x] **ENG-01**: Reduce repository-wide Ruff violations and expand CI lint scope.
- [x] **ENG-02**: Reduce core-module mypy debt and expand CI type-check scope.
- [x] **OPS-01**: Stabilize Beads backend/daemon behavior for consistent local tracking commands.

## v0.4 Typing Hardening Tranche (Completed)

- [x] **TYPE-01**: Clear mypy errors across full `src/asl_video_generator` module scope.
- [x] **TYPE-02**: Expand CI mypy gate to run against full package module tree.

## v0.5 Optional Backend Hardening (Completed)

- [x] **PRY-01**: Route `mesh_backend=pyrender` explicitly and detect optional dependency availability.
- [x] **PRY-02**: Fall back deterministically to `software_3d` when pyrender dependencies are unavailable.
- [x] **PRY-03**: Add regression tests and operator docs for pyrender fallback behavior.

## v0.6 Native Pyrender Backend MVP (Completed)

- [x] **NPY-01**: Implement native offscreen pyrender rendering path for mesh frames.
- [x] **NPY-02**: Keep deterministic software fallback when pyrender render path raises runtime errors.
- [x] **NPY-03**: Add deterministic unit tests and optional dependency docs/packaging for native pyrender mode.

## Out of Scope (Current Milestone)

- Re-architecting the entire pipeline into new infrastructure.
- Product-level mobile UX changes in consuming apps.
- Multi-tenant SaaS packaging or account systems.

## Traceability

| Requirement | Phase | Beads |
|------------|-------|-------|
| PROD-01 | Phase 3 | asl-video-generator-cxv.1 |
| PROD-02 | Phase 3 | asl-video-generator-cxv.2 |
| PROD-03 | Phase 3 | asl-video-generator-cxv.3 |
| PROD-04 | Phase 4 | asl-video-generator-cxv.5 |
| PROD-05 | Phase 3 | asl-video-generator-cxv.4 |
| FID-01 | Phase 5 | asl-video-generator-v5u.1 |
| FID-02 | Phase 5 | asl-video-generator-v5u.2 |
| FID-03 | Phase 5 | asl-video-generator-v5u.3 |
| ENG-01 | Phase 6 | asl-video-generator-v5u.4 |
| ENG-02 | Phase 6 | asl-video-generator-v5u.5 |
| OPS-01 | Phase 6 | asl-video-generator-v5u.6 |
| TYPE-01 | Phase 7 | asl-video-generator-t9r.1, asl-video-generator-t9r.2, asl-video-generator-t9r.3 |
| TYPE-02 | Phase 7 | asl-video-generator-t9r.4 |
| PRY-01 | Phase 8 | asl-video-generator-q7m.1 |
| PRY-02 | Phase 8 | asl-video-generator-q7m.1 |
| PRY-03 | Phase 8 | asl-video-generator-q7m.2, asl-video-generator-q7m.3 |
| NPY-01 | Phase 9 | asl-video-generator-r6p.1 |
| NPY-02 | Phase 9 | asl-video-generator-r6p.1, asl-video-generator-r6p.2 |
| NPY-03 | Phase 9 | asl-video-generator-r6p.2, asl-video-generator-r6p.3 |
