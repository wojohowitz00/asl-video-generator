# Roadmap: ASL Video Generator

## Milestones

- ✅ **v0.1 Baseline Pipeline** — core implementation and content publishing landed (2026-02-01 to 2026-02-02)
- ✅ **v0.2 Production Mesh Pipeline** — implementation and validation complete (2026-02-17)
- ✅ **v0.3 Rendering Fidelity + Code Health** — complete (2026-02-17)
- ✅ **v0.4 Typing Hardening Tranche** — complete (2026-02-17)
- ✅ **v0.5 Optional Pyrender Backend Hardening** — complete (2026-02-17)
- ✅ **v0.6 Native Pyrender Backend MVP** — complete (2026-02-18)

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
