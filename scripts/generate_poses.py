#!/usr/bin/env python3
"""Batch generate ASL pose sequences from lesson sentences.

Usage:
    uv run python scripts/generate_poses.py --output ./output/poses --limit 10
"""

import argparse
import json
import os
from pathlib import Path

from asl_video_generator import LessonParser, PoseGenerator
from asl_video_generator.gloss_translator import GlossSequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ASL pose sequences from lesson sentences"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./output/poses"),
        help="Output directory for pose JSON files"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of sentences to process"
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        default=None,
        help="Filter by scenario name (e.g., 'House', 'Work')"
    )
    parser.add_argument(
        "--difficulty", "-d",
        type=str,
        choices=["beginner", "intermediate", "advanced"],
        default=None,
        help="Filter by difficulty level"
    )
    parser.add_argument(
        "--lessons-path",
        type=Path,
        default=Path(
            os.environ.get(
                "ASL_LESSONS_PATH",
                str(
                    Path(__file__).resolve().parents[1]
                    / "assets"
                    / "lessons"
                    / "500_Sentences_1769817458445.md"
                ),
            )
        ),
        help="Path to lesson sentences markdown file"
    )
    
    args = parser.parse_args()
    
    # Parse lessons
    print(f"Loading lessons from: {args.lessons_path}")
    lesson_parser = LessonParser(args.lessons_path)
    sentences = lesson_parser.to_json()
    
    # Apply filters
    if args.scenario:
        sentences = [s for s in sentences if args.scenario.lower() in s["scenario"].lower()]
    
    if args.difficulty:
        sentences = [s for s in sentences if s["difficulty"] == args.difficulty]
    
    if args.limit:
        sentences = sentences[:args.limit]
    
    print(f"Processing {len(sentences)} sentences...")
    
    # Generate poses
    generator = PoseGenerator()
    args.output.mkdir(parents=True, exist_ok=True)
    
    results = []
    for i, sentence in enumerate(sentences):
        # Keep this script runnable without LLM credentials by using a simple
        # uppercase tokenization as a stand-in for a gloss sequence.
        gloss_tokens = (
            sentence["text"]
            .upper()
            .replace("?", "")
            .replace("!", "")
            .replace(".", "")
            .replace(",", "")
            .replace("\u2019", "")
            .replace("'", "")
            .split()
        )
        gloss_seq = GlossSequence(
            english=sentence["text"],
            gloss=gloss_tokens,
            estimated_duration_ms=len(gloss_tokens) * 500,
            difficulty=sentence["difficulty"],
        )
        pose = generator.generate(gloss_seq)
        
        output_path = args.output / f"{sentence['id']}.json"
        pose.save(output_path)
        
        results.append({
            "id": sentence["id"],
            "text": sentence["text"],
            "gloss": pose.gloss,
            "scenario": sentence["scenario"],
            "difficulty": sentence["difficulty"],
            "frames": len(pose.frames),
            "duration_ms": pose.total_duration_ms,
            "missing_signs": pose.missing_signs,
            "path": str(output_path),
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(sentences)} sentences")
    
    # Save manifest
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"\nGenerated {len(results)} pose sequences")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
