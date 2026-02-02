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
