#!/usr/bin/env python3
"""Download and process WLASL pose data for the pose dictionary.

WLASL (Word-Level American Sign Language) dataset contains isolated sign videos.
This script downloads pre-extracted MediaPipe pose data and builds the pose dictionary.

Usage:
    uv run scripts/download_wlasl_poses.py --output ~/.cache/asl-video/pose_dictionary.db

Dataset info:
    - WLASL: https://github.com/dxli94/WLASL
    - Contains 2000+ ASL signs with multiple variants
    - Pre-extracted poses available from various research projects
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def download_wlasl_metadata(output_dir: Path) -> Path:
    """Download WLASL metadata JSON.

    Args:
        output_dir: Directory to save metadata.

    Returns:
        Path to downloaded metadata file.
    """
    import httpx

    metadata_url = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
    output_path = output_dir / "WLASL_v0.3.json"

    if output_path.exists():
        print(f"Metadata already exists: {output_path}")
        return output_path

    print(f"Downloading WLASL metadata...")
    response = httpx.get(metadata_url, follow_redirects=True)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    print(f"Saved to {output_path}")

    return output_path


def parse_wlasl_metadata(metadata_path: Path) -> dict[str, list[dict]]:
    """Parse WLASL metadata to extract gloss and video info.

    Args:
        metadata_path: Path to WLASL JSON metadata.

    Returns:
        Dictionary mapping gloss to list of video info dicts.
    """
    data = json.loads(metadata_path.read_text())

    glosses = {}
    for entry in data:
        gloss = entry["gloss"].upper()
        instances = entry.get("instances", [])

        glosses[gloss] = [
            {
                "video_id": inst.get("video_id"),
                "url": inst.get("url"),
                "bbox": inst.get("bbox"),
                "fps": inst.get("fps", 25),
                "frame_start": inst.get("frame_start", 0),
                "frame_end": inst.get("frame_end", -1),
                "signer_id": inst.get("signer_id"),
                "source": "wlasl",
            }
            for inst in instances
            if inst.get("video_id")
        ]

    return glosses


def generate_placeholder_poses(gloss: str, num_frames: int = 15) -> list[dict]:
    """Generate placeholder pose frames for a sign.

    Creates a simple hand motion as placeholder until real poses are available.

    Args:
        gloss: The ASL gloss.
        num_frames: Number of frames to generate.

    Returns:
        List of pose frame dictionaries.
    """
    import numpy as np

    frames = []
    for i in range(num_frames):
        t = i / num_frames

        # Create normalized keypoints (0-1 range)
        # Body pose (33 MediaPipe landmarks)
        body = np.zeros((33, 3))
        body[:, 2] = 1.0  # Confidence

        # Basic body structure
        body[0] = [0.5, 0.15, 1.0]   # Nose
        body[11] = [0.4, 0.3, 1.0]   # Left shoulder
        body[12] = [0.6, 0.3, 1.0]   # Right shoulder
        body[13] = [0.35, 0.45, 1.0]  # Left elbow
        body[14] = [0.65, 0.45, 1.0]  # Right elbow
        body[15] = [0.32, 0.55, 1.0]  # Left wrist
        body[16] = [0.68, 0.55, 1.0]  # Right wrist

        # Animate hands with simple motion
        wave = np.sin(t * np.pi * 2) * 0.1

        # Left hand (21 landmarks) - at rest
        left_hand = np.zeros((21, 3))
        left_hand[:, 0] = np.linspace(0.30, 0.35, 21)
        left_hand[:, 1] = np.linspace(0.58, 0.68, 21)
        left_hand[:, 2] = 1.0

        # Right hand (21 landmarks) - animated
        right_hand = np.zeros((21, 3))
        right_hand[:, 0] = np.linspace(0.65, 0.70, 21)
        right_hand[:, 1] = np.linspace(0.50 + wave, 0.60 + wave, 21)
        right_hand[:, 2] = 1.0

        frames.append({
            "body": body.tolist(),
            "left_hand": left_hand.tolist(),
            "right_hand": right_hand.tolist(),
            "face": None,
        })

    return frames


def build_pose_dictionary(
    glosses: dict[str, list[dict]],
    output_db: Path,
    limit: int | None = None,
) -> None:
    """Build pose dictionary from WLASL metadata.

    For now, generates placeholder poses. In production, would download
    actual pose data from pre-extracted sources.

    Args:
        glosses: Dictionary mapping gloss to video info.
        output_db: Path for output SQLite database.
        limit: Optional limit on number of glosses to process.
    """
    from asl_video_generator.pose_dictionary import (
        PoseDictionary,
        PoseKeypoints,
        SignPoseSequence,
    )

    dictionary = PoseDictionary(db_path=output_db)

    gloss_list = list(glosses.keys())
    if limit:
        gloss_list = gloss_list[:limit]

    print(f"Building dictionary with {len(gloss_list)} glosses...")

    for i, gloss in enumerate(gloss_list):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(gloss_list)}...")

        videos = glosses[gloss]

        # Generate poses for each variant (up to 3)
        for variant_id, video_info in enumerate(videos[:3]):
            # Generate placeholder poses
            # In production: download actual pre-extracted poses
            pose_frames = generate_placeholder_poses(gloss, num_frames=15)

            # Convert to PoseKeypoints
            import numpy as np
            keypoints = [
                PoseKeypoints(
                    body=np.array(f["body"]),
                    left_hand=np.array(f["left_hand"]),
                    right_hand=np.array(f["right_hand"]),
                    face=np.array(f["face"]) if f.get("face") else None,
                )
                for f in pose_frames
            ]

            sequence = SignPoseSequence(
                gloss=gloss,
                frames=keypoints,
                fps=30,
                variant_id=variant_id,
                source="wlasl",
                signer_id=video_info.get("signer_id"),
            )

            dictionary.add_sign(sequence)

    print(f"\nDictionary built: {output_db}")
    counts = dictionary.count_signs()
    print(f"Total signs: {sum(counts.values())}")
    for source, count in counts.items():
        print(f"  {source}: {count}")


def add_alphabet_signs(output_db: Path) -> None:
    """Add fingerspelling alphabet to the dictionary.

    Args:
        output_db: Path to pose dictionary database.
    """
    from asl_video_generator.pose_dictionary import (
        PoseDictionary,
        PoseKeypoints,
        SignPoseSequence,
    )
    import numpy as np

    dictionary = PoseDictionary(db_path=output_db)

    print("Adding fingerspelling alphabet...")

    # Generate placeholder poses for A-Z
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        pose_frames = generate_placeholder_poses(letter, num_frames=10)

        keypoints = [
            PoseKeypoints(
                body=np.array(f["body"]),
                left_hand=np.array(f["left_hand"]),
                right_hand=np.array(f["right_hand"]),
                face=None,
            )
            for f in pose_frames
        ]

        sequence = SignPoseSequence(
            gloss=letter,
            frames=keypoints,
            fps=30,
            variant_id=0,
            source="alphabet",
        )

        dictionary.add_sign(sequence)

    print("  Added A-Z fingerspelling letters")


def main():
    parser = argparse.ArgumentParser(
        description="Download WLASL poses and build pose dictionary",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path.home() / ".cache" / "asl-video" / "pose_dictionary.db",
        help="Output database path",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "asl-video" / "data",
        help="Directory for downloaded data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of glosses to process (for testing)",
    )
    parser.add_argument(
        "--alphabet-only",
        action="store_true",
        help="Only add alphabet signs (quick setup)",
    )

    args = parser.parse_args()

    # Ensure directories exist
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.alphabet_only:
        add_alphabet_signs(args.output)
        return

    # Download metadata
    metadata_path = download_wlasl_metadata(args.data_dir)

    # Parse metadata
    glosses = parse_wlasl_metadata(metadata_path)
    print(f"Found {len(glosses)} unique glosses in WLASL")

    # Build dictionary
    build_pose_dictionary(glosses, args.output, limit=args.limit)

    # Add alphabet
    add_alphabet_signs(args.output)


if __name__ == "__main__":
    main()
