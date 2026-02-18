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

## v0.7 Render Script Routing Correctness (Completed)

- [x] **RND-01**: `scripts/render_videos.py` routes `--avatar-style mesh` runs to `AvatarRenderer.render_mesh`.
- [x] **RND-02**: Script routing behavior has regression tests for mesh vs. non-mesh avatar styles.

## v0.8 Stylized Routing Completion (Completed)

- [x] **STY-01**: `scripts/render_videos.py` routes `--avatar-style stylized` runs to `AvatarRenderer.render_mesh`.
- [x] **STY-02**: Regression tests cover stylized style routing to mesh renderer path.

## v0.9 Render3D CI Smoke Coverage (Completed)

- [x] **R3D-01**: Quality workflow includes dedicated `render3d` smoke job installing optional extras.
- [x] **R3D-02**: Repository includes render3d smoke tests validating optional dependency availability path.

## v1.0 Offscreen Smoke Execution (Completed)

- [x] **GLS-01**: Render3d smoke coverage includes actual offscreen pyrender render execution assertion.
- [x] **GLS-02**: Offscreen smoke assertion skips safely when GL context is unavailable in the environment.

## v1.1 Render Mesh End-to-End Smoke (Completed)

- [x] **E2E-01**: Render3d smoke suite covers `AvatarRenderer.render_mesh(...)` end-to-end output generation.
- [x] **E2E-02**: End-to-end smoke coverage remains compatible with optional render3d environment differences.

## v1.2 Pyrender Fallback Telemetry (Completed)

- [x] **TEL-01**: AvatarRenderer exposes runtime telemetry for effective mesh backend usage.
- [x] **TEL-02**: Telemetry tracks pyrender fallback event count for dependency-unavailable and runtime-error paths.
- [x] **TEL-03**: Regression tests assert telemetry outcomes for pyrender success and fallback scenarios.

## v1.3 Pyrender Camera-Angle Regression (Completed)

- [x] **CAM-01**: Render3d smoke suite asserts pyrender backend output changes across different camera angles.
- [x] **CAM-02**: Camera-angle regression test skips safely when native pyrender path is unavailable at runtime.

## v1.4 Mesh Backend Benchmark Baseline (Completed)

- [x] **BEN-01**: Add benchmark script for software_3d and pyrender requested backends on fixed sample motion.
- [x] **BEN-02**: Benchmark report includes timing summary and effective backend telemetry fields.
- [x] **BEN-03**: Add regression tests for benchmark script parser defaults and report generation behavior.

## v1.5 Benchmark Baseline Artifact Trendability (Completed)

- [x] **ART-01**: Commit baseline benchmark artifact under `docs/benchmarks/`.
- [x] **ART-02**: Add artifact schema validation test to preserve report comparability over time.
- [x] **ART-03**: Document rerun and trend comparison workflow in README.

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
| RND-01 | Phase 10 | asl-dy5.1 |
| RND-02 | Phase 10 | asl-dy5.2 |
| STY-01 | Phase 11 | asl-o3d.1 |
| STY-02 | Phase 11 | asl-o3d.2 |
| R3D-01 | Phase 12 | asl-cwy.1, asl-cwy.2 |
| R3D-02 | Phase 12 | asl-cwy.2 |
| GLS-01 | Phase 13 | asl-dmv.1 |
| GLS-02 | Phase 13 | asl-dmv.1 |
| E2E-01 | Phase 14 | asl-1ib.1 |
| E2E-02 | Phase 14 | asl-1ib.1 |
| TEL-01 | Phase 15 | asl-6tl.1 |
| TEL-02 | Phase 15 | asl-6tl.1, asl-6tl.2 |
| TEL-03 | Phase 15 | asl-6tl.2 |
| CAM-01 | Phase 16 | asl-95a.1 |
| CAM-02 | Phase 16 | asl-95a.1 |
| BEN-01 | Phase 17 | asl-atw.1 |
| BEN-02 | Phase 17 | asl-atw.1, asl-atw.2 |
| BEN-03 | Phase 17 | asl-atw.2 |
| ART-01 | Phase 18 | asl-2la.1 |
| ART-02 | Phase 18 | asl-2la.2 |
| ART-03 | Phase 18 | asl-2la.3 |
