"""CLI for ASL Video Generator.

Provides commands for the full text-to-ASL video pipeline:
- generate: Full pipeline (text → gloss → poses → video)
- translate: Text to ASL gloss only
- pose: Gloss to skeletal poses only
- render: Poses to video only

Usage:
    uv run asl-generate "Hello, how are you?" --output hello.mp4
    uv run asl-translate "Where is the library?" --output library_gloss.json
    uv run asl-pose library_gloss.json --output library_poses.json
    uv run asl-render library_poses.json --output library.mp4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from .config import PipelineConfig


def main() -> None:
    """Main CLI entry point for full pipeline."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate ASL avatar videos from English text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline - text to video
  uv run asl-generate "Hello, how are you?" --output hello.mp4

  # With quality preset
  uv run asl-generate "Thank you" --quality preview --output thanks.mp4

  # Batch processing
  uv run asl-generate --batch sentences.txt --output-dir ./videos/

  # Specify LLM provider
  uv run asl-generate "Good morning" --provider ollama --output morning.mp4
        """,
    )

    parser.add_argument(
        "text",
        nargs="?",
        type=str,
        help="English text to translate to ASL video",
    )
    parser.add_argument(
        "--batch", "-b",
        type=Path,
        help="Path to file with sentences (one per line) for batch processing",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output.mp4"),
        help="Output video path (default: output.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch processing",
    )
    parser.add_argument(
        "--quality", "-q",
        choices=["preview", "medium", "quality"],
        default="medium",
        help="Quality preset (default: medium)",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "gemini", "ollama"],
        default="openai",
        help="LLM provider for gloss translation (default: openai)",
    )
    parser.add_argument(
        "--render-mode", "-r",
        choices=["diffusion", "skeleton", "auto"],
        default="auto",
        help="Video rendering mode (default: auto)",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        help="Reference image for signer appearance (diffusion mode)",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate gloss and pose files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--check-device",
        action="store_true",
        help="Check device availability and exit",
    )

    args = parser.parse_args()

    # Handle device check
    if args.check_device:
        _check_device()
        return

    # Validate input
    if not args.text and not args.batch:
        parser.print_help()
        print("\nError: Please provide text or --batch file")
        sys.exit(1)

    # Import here to avoid slow startup for help
    from .config import QualityPreset, load_config_from_env

    config = load_config_from_env()
    config.quality = QualityPreset(args.quality)
    config.llm_provider = args.provider

    if args.batch:
        _run_batch(args, config)
    else:
        _run_single(args, config)


def _check_device() -> None:
    """Check and display device availability."""
    from .config import get_memory_info, validate_mps_availability

    print("Device Check")
    print("=" * 40)

    device_info = validate_mps_availability()
    print(f"Device: {device_info['device']}")
    print(f"Available: {device_info['available']}")

    if device_info.get("info"):
        print("\nDetails:")
        for key, value in device_info["info"].items():
            print(f"  {key}: {value}")

    if device_info.get("warnings"):
        print("\nWarnings:")
        for warning in device_info["warnings"]:
            print(f"  - {warning}")

    print("\nMemory:")
    mem_info = get_memory_info()
    for key, value in mem_info.items():
        print(f"  {key}: {value}")


def _run_single(args: argparse.Namespace, config: "PipelineConfig") -> None:
    """Run pipeline for a single sentence."""
    from .diffusion_renderer import DiffusionRenderer
    from .gloss_translator import GlossTranslator
    from .pose_generator import PoseGenerator

    text = args.text
    output_path = args.output

    print(f"Processing: {text}")

    # Stage 1: Translate to gloss
    print("\n[1/3] Translating to ASL gloss...")
    translator = GlossTranslator(provider=config.llm_provider, config=config)
    gloss_seq = translator.translate(text)

    print(f"  Gloss: {' '.join(gloss_seq.gloss)}")
    print(f"  Question: {gloss_seq.nmm.is_question} ({gloss_seq.nmm.question_type})")
    print(f"  Negation: {gloss_seq.nmm.is_negation}")

    if args.save_intermediate:
        gloss_path = output_path.with_suffix(".gloss.json")
        gloss_path.write_text(gloss_seq.model_dump_json(indent=2))
        print(f"  Saved: {gloss_path}")

    # Stage 2: Generate poses
    print("\n[2/3] Generating skeletal poses...")
    pose_gen = PoseGenerator(config=config)
    pose_seq = pose_gen.generate(gloss_seq)

    print(f"  Frames: {len(pose_seq.frames)}")
    print(f"  Duration: {pose_seq.total_duration_ms}ms")
    if pose_seq.missing_signs:
        print(f"  Missing signs: {pose_seq.missing_signs}")

    if args.save_intermediate:
        pose_path = output_path.with_suffix(".poses.json")
        pose_seq.save(pose_path)
        print(f"  Saved: {pose_path}")

    # Stage 3: Render video
    print("\n[3/3] Rendering video...")
    renderer = DiffusionRenderer(
        config=config,
        reference_image=args.reference_image,
    )
    result = renderer.render(
        pose_seq,
        output_path,
        mode=args.render_mode,
    )

    print("\nComplete!")
    print(f"  Output: {result.video_path}")
    print(f"  Resolution: {result.width}x{result.height}")
    print(f"  FPS: {result.fps}")
    print(f"  Mode: {result.render_mode}")


def _run_batch(args: argparse.Namespace, config: "PipelineConfig") -> None:
    """Run pipeline for batch of sentences."""
    from .diffusion_renderer import DiffusionRenderer
    from .gloss_translator import GlossTranslator
    from .pose_generator import PoseGenerator

    batch_file = args.batch
    output_dir = args.output_dir or Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read sentences
    sentences = batch_file.read_text().strip().split("\n")
    sentences = [s.strip() for s in sentences if s.strip()]

    print(f"Processing {len(sentences)} sentences...")

    translator = GlossTranslator(provider=config.llm_provider, config=config)
    pose_gen = PoseGenerator(config=config)
    renderer = DiffusionRenderer(config=config)

    for i, text in enumerate(sentences):
        print(f"\n[{i+1}/{len(sentences)}] {text[:50]}...")

        try:
            # Translate
            gloss_seq = translator.translate(text)

            # Generate poses
            pose_seq = pose_gen.generate(gloss_seq)

            # Render video
            output_path = output_dir / f"video_{i+1:03d}.mp4"
            result = renderer.render(
                pose_seq,
                output_path,
                mode=args.render_mode,
            )

            print(f"  -> {result.video_path}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nBatch complete! Videos saved to {output_dir}")


def translate_cmd() -> None:
    """CLI entry point for translate-only command."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Translate English text to ASL gloss",
    )
    parser.add_argument(
        "text",
        type=str,
        help="English text to translate",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSON file (optional, prints to stdout if not specified)",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["openai", "gemini", "ollama"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    from .config import load_config_from_env
    from .gloss_translator import GlossTranslator

    config = load_config_from_env()
    config.llm_provider = args.provider

    translator = GlossTranslator(provider=args.provider, config=config)
    result = translator.translate(args.text)

    if args.format == "json":
        output = result.model_dump_json(indent=2)
    else:
        output = f"English: {result.english}\n"
        output += f"Gloss: {' '.join(result.gloss)}\n"
        output += f"Duration: {result.estimated_duration_ms}ms\n"
        if result.nmm.is_question:
            output += f"Question type: {result.nmm.question_type}\n"
        if result.nmm.is_negation:
            output += "Negation: yes\n"

    if args.output:
        args.output.write_text(output)
        print(f"Saved to {args.output}")
    else:
        print(output)


def pose_cmd() -> None:
    """CLI entry point for pose-only command."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate skeletal poses from ASL gloss",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input gloss JSON file or gloss string",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output pose JSON file",
    )
    parser.add_argument(
        "--gloss", "-g",
        type=str,
        help="Direct gloss input (space-separated, e.g., 'HELLO HOW YOU')",
    )

    args = parser.parse_args()

    from .config import load_config_from_env
    from .gloss_translator import GlossSequence
    from .pose_generator import PoseGenerator

    config = load_config_from_env()
    pose_gen = PoseGenerator(config=config)

    # Load or parse gloss
    if args.gloss:
        gloss_seq = GlossSequence(
            english="",
            gloss=args.gloss.upper().split(),
            estimated_duration_ms=len(args.gloss.split()) * 500,
        )
    elif args.input.exists():
        data = json.loads(args.input.read_text())
        gloss_seq = GlossSequence(**data)
    else:
        # Treat input as gloss string
        gloss_seq = GlossSequence(
            english="",
            gloss=str(args.input).upper().split(),
            estimated_duration_ms=len(str(args.input).split()) * 500,
        )

    # Generate poses
    pose_seq = pose_gen.generate(gloss_seq)

    # Save
    pose_seq.save(args.output)
    print(f"Generated {len(pose_seq.frames)} frames")
    print(f"Duration: {pose_seq.total_duration_ms}ms")
    if pose_seq.missing_signs:
        print(f"Missing signs: {pose_seq.missing_signs}")
    print(f"Saved to {args.output}")


def render_cmd() -> None:
    """CLI entry point for render-only command."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Render video from skeletal poses",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input pose JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output video file",
    )
    parser.add_argument(
        "--quality", "-q",
        choices=["preview", "medium", "quality"],
        default="medium",
        help="Quality preset (default: medium)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["diffusion", "skeleton", "auto"],
        default="auto",
        help="Rendering mode (default: auto)",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        help="Reference image for signer appearance",
    )

    args = parser.parse_args()

    from .config import QualityPreset, load_config_from_env
    from .diffusion_renderer import render_from_pose_file

    config = load_config_from_env()
    config.quality = QualityPreset(args.quality)

    result = render_from_pose_file(
        args.input,
        args.output,
        config=config,
        mode=args.mode,
    )

    print(f"Rendered {result.num_frames} frames")
    print(f"Resolution: {result.width}x{result.height}")
    print(f"Mode: {result.render_mode}")
    print(f"Saved to {result.video_path}")


if __name__ == "__main__":
    main()
