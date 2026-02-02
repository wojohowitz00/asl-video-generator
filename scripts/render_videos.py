#!/usr/bin/env python3
"""Batch render ASL pose sequences to video.

Usage:
    uv run python scripts/render_videos.py --input ./output/poses --output ./output/videos
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig


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
    
    # Pose manifest contains curriculum metadata (scenario/difficulty/text) and
    # optionally gloss tokens. We'll merge it into a video manifest that matches
    # the learning app's expected schema.
    manifest_path = args.input / "manifest.json"
    manifest_items: list[dict] = []
    if manifest_path.exists():
        raw = json.loads(manifest_path.read_text())
        if isinstance(raw, list):
            manifest_items = raw
        elif isinstance(raw, dict) and isinstance(raw.get("items"), list):
            manifest_items = raw["items"]
    
    # Render videos
    renderer = AvatarRenderer(config)
    args.output.mkdir(parents=True, exist_ok=True)
    
    rendered_count = 0
    updated_items: list[dict] = []
    
    # Map ID to manifest entry
    manifest_map = {item.get("id"): item for item in manifest_items if item.get("id")}
    
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

            base = manifest_map.get(pose_id, {})

            # Fallback to reading the pose JSON for fields if the pose manifest
            # doesn't include them (older runs).
            pose_meta: dict = {}
            if not base.get("text") or not base.get("duration_ms") or not base.get("gloss"):
                try:
                    pose_meta = json.loads(json_path.read_text())
                except Exception:
                    pose_meta = {}

            english_text = (
                base.get("text")
                or base.get("englishText")
                or pose_meta.get("english")
                or pose_id
            )
            gloss = base.get("gloss") or pose_meta.get("gloss") or []
            duration_ms = (
                base.get("duration_ms")
                or base.get("durationMs")
                or pose_meta.get("total_duration_ms")
                or 0
            )

            item_version = "1.0.0"
            updated_items.append(
                {
                    "id": pose_id,
                    "type": "video",
                    # Filled in by scripts/upload_to_s3.py
                    "remoteUrl": "",
                    "scenario": base.get("scenario") or "unknown",
                    "difficulty": base.get("difficulty") or "beginner",
                    "englishText": english_text,
                    "gloss": gloss,
                    "durationMs": duration_ms,
                    "sizeBytes": output_path.stat().st_size if output_path.exists() else None,
                    "version": item_version,
                    # Local-only fields (stripped before publishing)
                    "localVideoPath": str(output_path),
                }
            )
                
        except Exception as e:
            print(f"Error rendering {pose_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save updated manifest to video output dir
    video_manifest_path = args.output / "manifest.json"
    now = datetime.now(timezone.utc).isoformat()
    video_manifest_content = {
        "version": "1.0.0",
        "updatedAt": now,
        "totalItems": len(updated_items),
        "scenarios": sorted({item.get("scenario") for item in updated_items if item.get("scenario")}),
        "items": updated_items,
    }
    video_manifest_path.write_text(json.dumps(video_manifest_content, indent=2))
    
    print(f"\nRendered {rendered_count} videos")
    print(f"Video manifest saved to: {video_manifest_path}")


if __name__ == "__main__":
    main()
