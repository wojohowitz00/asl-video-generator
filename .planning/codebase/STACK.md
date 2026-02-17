# Stack

Analysis Date: 2026-02-17

## Runtime

- Python >= 3.11
- Package/build via Hatchling
- Dependency management via uv

## Core Dependencies

- ML/model: torch, diffusers, transformers, accelerate
- Translation/providers: openai, google-generativeai, ollama
- Media: pillow, imageio[ffmpeg], opencv-python
- Data/modeling: pydantic, pydantic-settings, sqlalchemy, numpy, scipy

## Dev Tooling

- pytest
- ruff
- mypy
