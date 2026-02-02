#!/usr/bin/env python3
"""Batch render ASL pose sequences to video.

Usage:
    uv run python scripts/render_videos.py --input ./output/poses --output ./output/videos
"""

import argparse
import json
from pathlib import Path

from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig, render_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render ASL pose sequences to video"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("./output/poses"),
        help="Input directory containing pose JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./output/videos"),
        help="Output directory for rendered videos"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["mp4", "gif", "frames"],
        default="mp4",
        help="Output format"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Video width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Video height"
    )

    args = parser.parse_args()

    print(f"Rendering videos from {args.input} to {args.output}...")
    
    config = RenderConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        output_format=args.format,
        avatar_style="skeleton",  # Default to skeleton for now
        background_color=(255, 255, 255)
    )
    
    # Process manifest to keep track of rendered files
    manifest_path = args.input / "manifest.json"
    manifest_data = []
    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text())
    
    # Render videos
    renderer = AvatarRenderer(config)
    args.output.mkdir(parents=True, exist_ok=True)
    
    rendered_count = 0
    updated_manifest = []
    
    # Map ID to manifest entry
    manifest_map = {item["id"]: item for item in manifest_data}
    
    for json_path in args.input.glob("*.json"):
        if json_path.name == "manifest.json":
            continue
            
        pose_id = json_path.stem
        output_path = args.output / f"{pose_id}.{args.format}"
        
        # Skip if already exists (optional, maybe add force flag)
        # if output_path.exists():
        #     print(f"Skipping {pose_id} (already exists)")
        #     continue
            
        try:
            print(f"Rendering {pose_id}...")
            renderer.render_poses(json_path, output_path)
            rendered_count += 1
            
            # Update manifest entry with video path
            if pose_id in manifest_map:
                entry = manifest_map[pose_id]
                entry["video_path"] = str(output_path)
                entry["video_url"] = f"https://cdn.example.com/asl-content/{pose_id}.{args.format}" # Placeholder
                updated_manifest.append(entry)
                
        except Exception as e:
            print(f"Error rendering {pose_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save updated manifest to video output dir
    video_manifest_path = args.output / "manifest.json"
    video_manifest_content = {
        "version": "1.0.0",
        "updatedAt": "2024-05-23T12:00:00Z", # Should be dynamic
        "totalItems": len(updated_manifest),
        "scenarios": list(set(item["scenario"] for item in updated_manifest if "scenario" in item)),
        "items": updated_manifest
    }
    video_manifest_path.write_text(json.dumps(video_manifest_content, indent=2))
    
    print(f"\nRendered {rendered_count} videos")
    print(f"Video manifest saved to: {video_manifest_path}")


if __name__ == "__main__":
    main()
