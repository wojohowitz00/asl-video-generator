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
- [ ] **QUAL-02**: Automated CI validates every change.
- [ ] **QUAL-03**: Critical pipeline paths have integration coverage.

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
