# Structure

Analysis Date: 2026-02-17

## Top-Level Layout

- `src/asl_video_generator/` - core package implementation
- `scripts/` - operational/generation/upload scripts
- `tests/` - pytest suites
- `assets/` - curriculum and source content
- `output/` - generated artifacts
- `docs/` - schemas and supporting docs

## Key Modules

- `gloss_translator.py` - text to ASL gloss
- `pose_generator.py` - gloss to poses
- `vocabulary_generator.py` - word-level mesh motion generation (placeholder sections exist)
- `avatar_renderer.py` - pose/mesh rendering (mesh path partially placeholder)
- `diffusion_renderer.py` - rendering backend
- `lesson_parser.py` - curriculum parsing
- `cli.py` - command entrypoints and orchestration
