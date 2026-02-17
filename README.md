# ASL Video Generator

Cloud pipeline for generating photorealistic ASL avatar videos from English text.

## Architecture

```
English Text → LLM (GPT-4o/Gemini) → ASL Gloss → SignGen → Video
```

## Setup

```bash
# Install dependencies
uv sync

# Full pipeline (text -> gloss -> poses -> video)
uv run asl-generate "Hello, how are you?" --output ./output/hello.mp4

# Translate only
uv run asl-translate "Where is the library?" --format json --output ./output/library.gloss.json

# Pose generation only
uv run asl-pose ./output/library.gloss.json --output ./output/library.poses.json

# Render only
uv run asl-render ./output/library.poses.json --output ./output/library.mp4 --mode skeleton
```

## Curriculum Source

- Default curriculum file: `assets/lessons/500_Sentences_1769817458445.md`
- Override with: `ASL_LESSONS_PATH=/path/to/500_Sentences_....md`

## Publishing Content For The Learning App

1. Generate poses: `uv run python scripts/generate_poses.py --output ./output/poses`
2. Render videos: `uv run python scripts/render_videos.py --input ./output/poses --output ./output/videos`
3. Upload + publish manifest: `uv run python scripts/upload_to_s3.py --input ./output/videos`

The learning app should set `EXPO_PUBLIC_CONTENT_MANIFEST_URL` to the uploaded
`manifest.json` URL.

## Components

- `gloss_translator.py` - LLM-based English→ASL gloss translation
- `lesson_parser.py` - Parse 500 lesson sentences from curriculum
- `pose_generator.py` - Generate pose sequences from ASL gloss
- `diffusion_renderer.py` - Render videos from pose sequences
- `avatar_renderer.py` - Render skeletal/mesh animations to video
- `cli.py` - Command-line interface

## Usage

```python
from asl_video_generator import generate_asl_video

# Generate a video from English text
video_path = generate_asl_video(
    "Good morning!",
    output_path="good_morning.mp4",
    provider="openai",
    render_mode="skeleton",
)
print(video_path)
```
