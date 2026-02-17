# Roadmap: ASL Video Generator

## Milestones

- ✅ **v0.1 Baseline Pipeline** — core implementation and content publishing landed (2026-02-01 to 2026-02-02)
- ✅ **v0.2 Production Mesh Pipeline** — implementation and validation complete (2026-02-17)

## Phases

- [x] **Phase 1: Core Pipeline Foundation** - Python package, CLI, gloss/pose pipeline
- [x] **Phase 2: Content Publishing Workflow** - curriculum in-repo + pose/video/manifest scripts
- [ ] **Phase 3: Mesh Productionization** - real model/rendering + docs alignment
- [ ] **Phase 4: Reliability Hardening** - CI quality gates + broader regression coverage

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
- [ ] `asl-video-generator-cxv.5` - Add CI quality gates for pytest, ruff, and mypy

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Pipeline Foundation | baseline delivered | Complete | 2026-02-01 |
| 2. Content Publishing Workflow | baseline delivered | Complete | 2026-02-02 |
| 3. Mesh Productionization | 4/4 | Complete | 2026-02-17 |
| 4. Reliability Hardening | 1/1 | Complete | 2026-02-17 |
