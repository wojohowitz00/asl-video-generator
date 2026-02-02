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

# Run the generator
uv run generate-asl --input "Hello, how are you?"
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
- `video_generator.py` - SignGen integration for avatar video generation
- `cli.py` - Command-line interface

## Usage

```python
from asl_video_generator import GlossTranslator, VideoGenerator

# Translate English to ASL gloss
translator = GlossTranslator()
gloss = translator.translate("Good morning!")
# Output: {"gloss": ["GOOD", "MORNING"], "nmm": {"facial": "neutral"}}

# Generate video
generator = VideoGenerator()
video_path = generator.generate(gloss)
```
