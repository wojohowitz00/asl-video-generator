# Architecture

Analysis Date: 2026-02-17

## Pattern Overview

Overall: Python package with CLI orchestration for a staged generation pipeline.

Primary flow:
1. English text -> gloss translation (`gloss_translator.py`)
2. Gloss -> pose/motion generation (`pose_generator.py`, `vocabulary_generator.py`)
3. Pose/motion -> rendered media (`diffusion_renderer.py`, `avatar_renderer.py`)
4. Content publishing scripts -> manifest output (`scripts/*.py`)

## Layers

- CLI layer: `src/asl_video_generator/cli.py`
- Domain pipeline: translators/generators/renderers under `src/asl_video_generator/`
- Publishing utilities: scripts in `scripts/`
- Test layer: pytest suites in `tests/`

## Entry Points

- `asl-generate` - full pipeline
- `asl-translate` - translation only
- `asl-pose` - pose generation only
- `asl-render` - rendering only

Defined in `pyproject.toml`.
