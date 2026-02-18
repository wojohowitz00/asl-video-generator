# Roadmap: ASL Video Generator

## Milestones

- ✅ **v0.1 Baseline Pipeline** — core implementation and content publishing landed (2026-02-01 to 2026-02-02)
- ✅ **v0.2 Production Mesh Pipeline** — implementation and validation complete (2026-02-17)
- ✅ **v0.3 Rendering Fidelity + Code Health** — complete (2026-02-17)
- ✅ **v0.4 Typing Hardening Tranche** — complete (2026-02-17)
- ✅ **v0.5 Optional Pyrender Backend Hardening** — complete (2026-02-17)
- ✅ **v0.6 Native Pyrender Backend MVP** — complete (2026-02-18)
- ✅ **v0.7 Render Script Routing Correctness** — complete (2026-02-18)
- ✅ **v0.8 Stylized Routing Completion** — complete (2026-02-18)
- ✅ **v0.9 Render3D CI Smoke Coverage** — complete (2026-02-18)
- ✅ **v1.0 Offscreen Smoke Execution** — complete (2026-02-18)
- ✅ **v1.1 Render Mesh End-to-End Smoke** — complete (2026-02-18)
- ✅ **v1.2 Pyrender Fallback Telemetry** — complete (2026-02-18)
- ✅ **v1.3 Pyrender Camera-Angle Regression** — complete (2026-02-18)
- ✅ **v1.4 Mesh Backend Benchmark Baseline** — complete (2026-02-18)

## Phases

- [x] **Phase 1: Core Pipeline Foundation** - Python package, CLI, gloss/pose pipeline
- [x] **Phase 2: Content Publishing Workflow** - curriculum in-repo + pose/video/manifest scripts
- [x] **Phase 3: Mesh Productionization** - real model/rendering + docs alignment
- [x] **Phase 4: Reliability Hardening** - CI quality gates + broader regression coverage
- [x] **Phase 5: Rendering Fidelity Upgrade** - true 3D mesh rasterization + fidelity controls
- [x] **Phase 6: Repository Health Hardening** - lint/type debt reduction + tracking stability
- [x] **Phase 7: Typing Hardening Tranche** - full-source mypy cleanup + CI type gate expansion
- [x] **Phase 8: Optional Backend Hardening** - pyrender routing/fallback robustness + docs
- [x] **Phase 9: Native Pyrender Backend MVP** - native offscreen pyrender render path + runtime fallback safety
- [x] **Phase 10: Render Script Routing Correctness** - mesh avatar-style batch runs invoke mesh renderer path
- [x] **Phase 11: Stylized Routing Completion** - stylized avatar-style batch runs invoke mesh renderer path
- [x] **Phase 12: Render3D CI Smoke Coverage** - optional render3d dependency path validated in CI
- [x] **Phase 13: Offscreen Smoke Execution** - render3d smoke includes real offscreen render assertion
- [x] **Phase 14: Render Mesh End-to-End Smoke** - render3d smoke covers end-to-end render_mesh output generation
- [x] **Phase 15: Pyrender Fallback Telemetry** - effective backend and fallback counts are observable at runtime
- [x] **Phase 16: Pyrender Camera-Angle Regression** - render3d smoke validates camera-angle sensitivity for pyrender outputs
- [x] **Phase 17: Mesh Backend Benchmark Baseline** - reproducible backend runtime benchmark script and report are available

## Phase Details

### Phase 3: Mesh Productionization
**Goal**: Convert placeholder mesh/model paths into production-ready flows.
**Depends on**: Phase 2 complete
**Requirements**: PROD-01, PROD-02, PROD-03, PROD-05
**Success Criteria**:
1. Vocabulary generation can run with real model-backed outputs.
2. Mesh rendering generates playable video outputs (not frame placeholders only).
3. E2E tests cover vocabulary -> mesh -> video behavior.
4. README commands and examples match actual CLI entry points.

Plans / linked Beads issues:
- [x] `asl-video-generator-cxv.1` - Implement wSignGen inference in VocabularyGenerator
- [x] `asl-video-generator-cxv.2` - Implement mesh video rendering pipeline in AvatarRenderer
- [x] `asl-video-generator-cxv.3` - Add end-to-end tests for vocabulary->mesh->video pipeline
- [x] `asl-video-generator-cxv.4` - Align README CLI examples with current entry points

### Phase 4: Reliability Hardening
**Goal**: Enforce quality checks in automation and reduce integration regressions.
**Depends on**: Phase 3 substantially complete
**Requirements**: PROD-04, QUAL-02, QUAL-03
**Success Criteria**:
1. CI runs pytest, ruff, and mypy on each change.
2. Failing quality checks are visible before merge.
3. Regression risk in core paths is reduced by automated gates.

Plans / linked Beads issues:
- [x] `asl-video-generator-cxv.5` - Add CI quality gates for pytest, ruff, and mypy

### Phase 5: Rendering Fidelity Upgrade
**Goal**: Move from stylized mesh visualizations to true 3D mesh rendering behavior.
**Depends on**: Phase 4 complete
**Requirements**: FID-01, FID-02, FID-03
**Success Criteria**:
1. Mesh rendering path uses an explicit 3D backend instead of coefficient-driven 2D proxy drawing.
2. CLI and scripts expose backend/style controls with sensible defaults.
3. Regression tests validate render artifacts for fidelity-relevant properties.

Plans / linked Beads issues:
- [x] `asl-video-generator-v5u.1` - Implement true 3D mesh rasterization backend
- [x] `asl-video-generator-v5u.2` - Add mesh rendering backend controls in CLI/scripts (depends on .1)
- [x] `asl-video-generator-v5u.3` - Add visual/integration tests for high-fidelity mesh rendering (depends on .1)

### Phase 6: Repository Health Hardening
**Goal**: Reduce global lint/type debt and expand CI gate scope while keeping developer workflows stable.
**Depends on**: Phase 4 complete (Phase 5 independent except where explicitly linked)
**Requirements**: ENG-01, ENG-02, OPS-01
**Success Criteria**:
1. Ruff violations are reduced enough to broaden lint gate beyond current targeted subset.
2. Mypy errors are reduced for core modules and CI type scope expands safely.
3. Beads backend/daemon config is stable across sessions with fewer operational warnings.

Plans / linked Beads issues:
- [x] `asl-video-generator-v5u.4` - Reduce repo-wide ruff debt and broaden lint gate
- [x] `asl-video-generator-v5u.5` - Reduce mypy debt for core modules and broaden type gate (depends on .4)
- [x] `asl-video-generator-v5u.6` - Stabilize beads backend config for this repo

### Phase 7: Typing Hardening Tranche
**Goal**: Remove remaining full-source mypy errors and broaden CI type checks to package-wide scope.
**Depends on**: Phase 6 complete
**Requirements**: TYPE-01, TYPE-02
**Success Criteria**:
1. `mypy --follow-imports=skip src/asl_video_generator` passes locally.
2. CI mypy step checks full package source scope.
3. Changes preserve existing runtime behavior and test pass status.

Plans / linked Beads issues:
- [x] `asl-video-generator-t9r.1` - Fix mypy typing debt in avatar_renderer + cli/__init__ glue
- [x] `asl-video-generator-t9r.2` - Fix mypy typing debt in gloss_translator
- [x] `asl-video-generator-t9r.3` - Fix mypy typing debt in diffusion_renderer + pose_generator
- [x] `asl-video-generator-t9r.4` - Expand CI mypy gate to include newly clean modules

### Phase 8: Optional Backend Hardening
**Goal**: Make optional pyrender backend behavior explicit, tested, and operator-friendly.
**Depends on**: Phase 5 complete
**Requirements**: PRY-01, PRY-02, PRY-03
**Success Criteria**:
1. `mesh_backend=pyrender` has explicit runtime routing path.
2. Missing pyrender dependencies trigger deterministic software fallback.
3. Tests and docs cover expected fallback behavior.

Plans / linked Beads issues:
- [x] `asl-video-generator-q7m.1` - Implement explicit pyrender backend routing and graceful fallback
- [x] `asl-video-generator-q7m.2` - Add regression tests for pyrender fallback and routing behavior
- [x] `asl-video-generator-q7m.3` - Document pyrender backend expectations in README/CLI help

### Phase 9: Native Pyrender Backend MVP
**Goal**: Replace placeholder pyrender delegation with a native offscreen render path while preserving deterministic fallback behavior.
**Depends on**: Phase 8 complete
**Requirements**: NPY-01, NPY-02, NPY-03
**Success Criteria**:
1. `mesh_backend=pyrender` executes native pyrender offscreen rendering when optional deps are present.
2. Runtime failures in native pyrender path fall back safely to `software_3d` without pipeline interruption.
3. Tests and operator docs cover native path behavior and optional `render3d` dependency install.

Plans / linked Beads issues:
- [x] `asl-video-generator-r6p.1` - Implement native pyrender offscreen render path in AvatarRenderer
- [x] `asl-video-generator-r6p.2` - Add unit tests for native pyrender path and fallback-on-error behavior
- [x] `asl-video-generator-r6p.3` - Add optional render3d dependency group + docs

### Phase 10: Render Script Routing Correctness
**Goal**: Ensure batch rendering script routes mesh-style jobs through the mesh renderer path.
**Depends on**: Phase 9 complete
**Requirements**: RND-01, RND-02
**Success Criteria**:
1. `--avatar-style mesh` in `scripts/render_videos.py` calls `AvatarRenderer.render_mesh`.
2. Non-mesh styles continue to use `AvatarRenderer.render_poses`.
3. Regression tests cover both routing paths.

Plans / linked Beads issues:
- [x] `asl-dy5.1` - Route mesh avatar-style runs to AvatarRenderer.render_mesh
- [x] `asl-dy5.2` - Add regression tests for render_videos style-based renderer routing
- [x] `asl-dy5.3` - Update planning docs for v0.7 routing fix tranche

### Phase 11: Stylized Routing Completion
**Goal**: Complete batch script routing behavior so stylized avatar style uses mesh rendering path.
**Depends on**: Phase 10 complete
**Requirements**: STY-01, STY-02
**Success Criteria**:
1. `--avatar-style stylized` in `scripts/render_videos.py` calls `AvatarRenderer.render_mesh`.
2. Existing mesh and skeleton routing behavior remains correct.
3. Regression tests cover stylized style route.

Plans / linked Beads issues:
- [x] `asl-o3d.1` - Route stylized avatar-style runs to AvatarRenderer.render_mesh
- [x] `asl-o3d.2` - Add regression test for stylized style routing
- [x] `asl-o3d.3` - Update planning docs for v0.8 routing completion

### Phase 12: Render3D CI Smoke Coverage
**Goal**: Continuously validate optional `render3d` dependency path in CI.
**Depends on**: Phase 9 complete
**Requirements**: R3D-01, R3D-02
**Success Criteria**:
1. Quality workflow defines a dedicated render3d smoke job with optional extras install.
2. Smoke tests validate optional dependency probe behavior for `pyrender` path.
3. A workflow guard test prevents accidental removal of render3d smoke coverage.

Plans / linked Beads issues:
- [x] `asl-cwy.1` - Add CI guard test for render3d smoke job in quality workflow
- [x] `asl-cwy.2` - Add optional-deps smoke test and wire render3d CI job
- [x] `asl-cwy.3` - Update planning docs for v0.9 CI smoke tranche

### Phase 13: Offscreen Smoke Execution
**Goal**: Extend render3d smoke coverage to include actual offscreen render execution behavior.
**Depends on**: Phase 12 complete
**Requirements**: GLS-01, GLS-02
**Success Criteria**:
1. Smoke suite includes an offscreen pyrender execution assertion.
2. The offscreen assertion skips safely on environments lacking GL context.
3. Full repo verification continues passing after smoke expansion.

Plans / linked Beads issues:
- [x] `asl-dmv.1` - Add safe-skip offscreen render smoke test
- [x] `asl-dmv.2` - Update planning docs for v1.0 offscreen smoke tranche

### Phase 14: Render Mesh End-to-End Smoke
**Goal**: Extend render3d smoke coverage to include `AvatarRenderer.render_mesh(...)` output artifact generation.
**Depends on**: Phase 13 complete
**Requirements**: E2E-01, E2E-02
**Success Criteria**:
1. Smoke tests generate a real output artifact through `render_mesh` with `mesh_backend=pyrender`.
2. Coverage remains stable across environments by relying on existing runtime fallback behavior.
3. Full repository quality gates remain green.

Plans / linked Beads issues:
- [x] `asl-1ib.1` - Add end-to-end render_mesh smoke test under render3d suite
- [x] `asl-1ib.2` - Update planning docs for v1.1 render_mesh smoke tranche

### Phase 15: Pyrender Fallback Telemetry
**Goal**: Add runtime observability for effective mesh backend selection and pyrender fallback behavior.
**Depends on**: Phase 9 complete
**Requirements**: TEL-01, TEL-02, TEL-03
**Success Criteria**:
1. AvatarRenderer exposes telemetry for last used backend and per-backend usage counts.
2. Pyrender fallback events are counted for unavailable dependencies and runtime render failures.
3. Regression tests validate telemetry outcomes across success and fallback paths.

Plans / linked Beads issues:
- [x] `asl-6tl.1` - Track effective mesh backend usage in AvatarRenderer
- [x] `asl-6tl.2` - Add regression tests for backend-usage telemetry
- [x] `asl-6tl.3` - Update planning docs for v1.2 telemetry tranche

### Phase 16: Pyrender Camera-Angle Regression
**Goal**: Add regression coverage proving pyrender backend artifacts respond to camera-angle changes.
**Depends on**: Phase 9 complete
**Requirements**: CAM-01, CAM-02
**Success Criteria**:
1. Smoke tests compare pyrender `render_mesh` outputs at multiple camera angles.
2. Camera-angle regression uses runtime-safe skip behavior when native path is unavailable.
3. Full repository quality gates remain green.

Plans / linked Beads issues:
- [x] `asl-95a.1` - Add pyrender camera-angle artifact regression test
- [x] `asl-95a.2` - Update planning docs for v1.3 camera-angle regression

### Phase 17: Mesh Backend Benchmark Baseline
**Goal**: Add reproducible local benchmarking for software_3d vs pyrender requested backends.
**Depends on**: Phase 15 complete
**Requirements**: BEN-01, BEN-02, BEN-03
**Success Criteria**:
1. A benchmark script runs both backend requests on fixed synthetic motion.
2. JSON report includes timing summary and effective backend telemetry fields.
3. Script behavior has automated regression coverage.

Plans / linked Beads issues:
- [x] `asl-atw.1` - Add benchmark_mesh_backends script
- [x] `asl-atw.2` - Add tests for benchmark script behavior
- [x] `asl-atw.3` - Document benchmark workflow + update planning

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Pipeline Foundation | baseline delivered | Complete | 2026-02-01 |
| 2. Content Publishing Workflow | baseline delivered | Complete | 2026-02-02 |
| 3. Mesh Productionization | 4/4 | Complete | 2026-02-17 |
| 4. Reliability Hardening | 1/1 | Complete | 2026-02-17 |
| 5. Rendering Fidelity Upgrade | 3/3 | Complete | 2026-02-17 |
| 6. Repository Health Hardening | 3/3 | Complete | 2026-02-17 |
| 7. Typing Hardening Tranche | 4/4 | Complete | 2026-02-17 |
| 8. Optional Backend Hardening | 3/3 | Complete | 2026-02-17 |
| 9. Native Pyrender Backend MVP | 3/3 | Complete | 2026-02-18 |
| 10. Render Script Routing Correctness | 3/3 | Complete | 2026-02-18 |
| 11. Stylized Routing Completion | 3/3 | Complete | 2026-02-18 |
| 12. Render3D CI Smoke Coverage | 3/3 | Complete | 2026-02-18 |
| 13. Offscreen Smoke Execution | 2/2 | Complete | 2026-02-18 |
| 14. Render Mesh End-to-End Smoke | 2/2 | Complete | 2026-02-18 |
| 15. Pyrender Fallback Telemetry | 3/3 | Complete | 2026-02-18 |
| 16. Pyrender Camera-Angle Regression | 2/2 | Complete | 2026-02-18 |
| 17. Mesh Backend Benchmark Baseline | 3/3 | Complete | 2026-02-18 |
