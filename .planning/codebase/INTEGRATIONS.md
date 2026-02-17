# Integrations

Analysis Date: 2026-02-17

## Model / AI Providers

- OpenAI (`openai`)
- Google Gemini (`google-generativeai`)
- Ollama (`ollama`)

## Media / ML Libraries

- PyTorch, diffusers, transformers, accelerate
- mediapipe, controlnet-aux
- pillow, imageio[ffmpeg], opencv-python

## Storage / Publishing

- Curriculum source file under `assets/lessons/`
- Output artifacts under `output/`
- Upload/publish workflow in `scripts/upload_to_s3.py`

## Config

- Environment variables loaded via `python-dotenv` in CLI.
