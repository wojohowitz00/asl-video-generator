# Roadmap: ASL Video Generator

## Milestones

- ✅ **v0.1 Baseline Pipeline** — core implementation and content publishing landed (2026-02-01 to 2026-02-02)
- ✅ **v0.2 Production Mesh Pipeline** — implementation and validation complete (2026-02-17)
- 🚧 **v0.3 Rendering Fidelity + Code Health** — planned next milestone (created 2026-02-17)

## Phases

- [x] **Phase 1: Core Pipeline Foundation** - Python package, CLI, gloss/pose pipeline
- [x] **Phase 2: Content Publishing Workflow** - curriculum in-repo + pose/video/manifest scripts
- [x] **Phase 3: Mesh Productionization** - real model/rendering + docs alignment
- [x] **Phase 4: Reliability Hardening** - CI quality gates + broader regression coverage
- [x] **Phase 5: Rendering Fidelity Upgrade** - true 3D mesh rasterization + fidelity controls
- [x] **Phase 6: Repository Health Hardening** - lint/type debt reduction + tracking stability

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

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Pipeline Foundation | baseline delivered | Complete | 2026-02-01 |
| 2. Content Publishing Workflow | baseline delivered | Complete | 2026-02-02 |
| 3. Mesh Productionization | 4/4 | Complete | 2026-02-17 |
| 4. Reliability Hardening | 1/1 | Complete | 2026-02-17 |
| 5. Rendering Fidelity Upgrade | 3/3 | Complete | 2026-02-17 |
| 6. Repository Health Hardening | 3/3 | Complete | 2026-02-17 |
