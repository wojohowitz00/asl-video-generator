"""Pose dictionary management for ASL sign lookup.

This module provides a dictionary-based approach to pose generation,
storing and retrieving pre-recorded pose sequences for ASL signs
from datasets like WLASL and How2Sign.
"""

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig, load_config_from_env


@dataclass
class PoseKeypoints:
    """Keypoints for a single frame in MediaPipe/OpenPose format."""

    # Body pose (33 MediaPipe landmarks or 25 OpenPose)
    body: np.ndarray  # Shape: (N, 3) for (x, y, confidence)
    # Left hand (21 landmarks)
    left_hand: np.ndarray  # Shape: (21, 3)
    # Right hand (21 landmarks)
    right_hand: np.ndarray  # Shape: (21, 3)
    # Face mesh (468 MediaPipe or 70 OpenPose) - optional
    face: np.ndarray | None = None  # Shape: (N, 3)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "body": self.body.tolist(),
            "left_hand": self.left_hand.tolist(),
            "right_hand": self.right_hand.tolist(),
            "face": self.face.tolist() if self.face is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseKeypoints":
        """Create from dictionary."""
        return cls(
            body=np.array(data["body"]),
            left_hand=np.array(data["left_hand"]),
            right_hand=np.array(data["right_hand"]),
            face=np.array(data["face"]) if data.get("face") else None,
        )

    def distance_to(self, other: "PoseKeypoints") -> float:
        """Calculate Euclidean distance to another pose (for motion matching).

        Uses weighted combination of body, left hand, and right hand distances.
        Hands are weighted more heavily as they carry most sign information.
        """
        # Focus on hands for ASL - they carry the most information
        left_dist = np.linalg.norm(self.left_hand[:, :2] - other.left_hand[:, :2])
        right_dist = np.linalg.norm(self.right_hand[:, :2] - other.right_hand[:, :2])
        # Body distance (less important for signs)
        body_dist = np.linalg.norm(self.body[:, :2] - other.body[:, :2])

        # Weight hands more heavily
        return float(0.4 * left_dist + 0.4 * right_dist + 0.2 * body_dist)


@dataclass
class SignPoseSequence:
    """A complete pose sequence for a single ASL sign."""

    gloss: str
    frames: list[PoseKeypoints]
    fps: int = 30
    duration_ms: int = 0
    variant_id: int = 0  # Different recordings of same sign
    source: str = "unknown"  # Dataset source (wlasl, how2sign, etc.)
    signer_id: str | None = None

    def __post_init__(self) -> None:
        if self.duration_ms == 0 and self.frames:
            self.duration_ms = int(len(self.frames) / self.fps * 1000)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "gloss": self.gloss,
            "frames": [f.to_dict() for f in self.frames],
            "fps": self.fps,
            "duration_ms": self.duration_ms,
            "variant_id": self.variant_id,
            "source": self.source,
            "signer_id": self.signer_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignPoseSequence":
        """Create from dictionary."""
        return cls(
            gloss=data["gloss"],
            frames=[PoseKeypoints.from_dict(f) for f in data["frames"]],
            fps=data.get("fps", 30),
            duration_ms=data.get("duration_ms", 0),
            variant_id=data.get("variant_id", 0),
            source=data.get("source", "unknown"),
            signer_id=data.get("signer_id"),
        )

    @property
    def first_frame(self) -> PoseKeypoints | None:
        """Get first frame for motion matching."""
        return self.frames[0] if self.frames else None

    @property
    def last_frame(self) -> PoseKeypoints | None:
        """Get last frame for motion matching."""
        return self.frames[-1] if self.frames else None


class PoseDictionary:
    """SQLite-backed dictionary of ASL sign poses.

    Stores pose sequences indexed by gloss, supporting:
    - Multiple variants per sign
    - Efficient lookup by gloss
    - Motion matching queries (find best transition)
    """

    def __init__(self, db_path: Path | None = None, config: PipelineConfig | None = None):
        """Initialize pose dictionary.

        Args:
            db_path: Path to SQLite database. If None, uses config default.
            config: Pipeline configuration.
        """
        self.config = config or load_config_from_env()
        self.db_path = db_path or (self.config.cache_dir / "pose_dictionary.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gloss TEXT NOT NULL,
                    variant_id INTEGER DEFAULT 0,
                    fps INTEGER DEFAULT 30,
                    duration_ms INTEGER,
                    source TEXT,
                    signer_id TEXT,
                    frames_json TEXT NOT NULL,
                    first_frame_json TEXT,
                    last_frame_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(gloss, variant_id, source)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signs_gloss ON signs(gloss)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signs_source ON signs(source)")

    def add_sign(self, sequence: SignPoseSequence) -> int:
        """Add a sign pose sequence to the dictionary.

        Args:
            sequence: The pose sequence to add.

        Returns:
            The database ID of the inserted record.
        """
        frames_json = json.dumps([f.to_dict() for f in sequence.frames])
        first_json = json.dumps(sequence.first_frame.to_dict()) if sequence.first_frame else None
        last_json = json.dumps(sequence.last_frame.to_dict()) if sequence.last_frame else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT OR REPLACE INTO signs
                   (gloss, variant_id, fps, duration_ms, source, signer_id,
                    frames_json, first_frame_json, last_frame_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sequence.gloss.upper(),
                    sequence.variant_id,
                    sequence.fps,
                    sequence.duration_ms,
                    sequence.source,
                    sequence.signer_id,
                    frames_json,
                    first_json,
                    last_json,
                ),
            )
            return cursor.lastrowid or 0

    def get_sign(self, gloss: str, variant_id: int = 0) -> SignPoseSequence | None:
        """Retrieve a specific sign variant.

        Args:
            gloss: The ASL gloss to look up.
            variant_id: Which variant to retrieve (default 0).

        Returns:
            SignPoseSequence if found, None otherwise.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT gloss, variant_id, fps, duration_ms, source, signer_id, frames_json
                   FROM signs WHERE gloss = ? AND variant_id = ?""",
                (gloss.upper(), variant_id),
            ).fetchone()

            if row:
                frames = [PoseKeypoints.from_dict(f) for f in json.loads(row[6])]
                return SignPoseSequence(
                    gloss=row[0],
                    variant_id=row[1],
                    fps=row[2],
                    duration_ms=row[3],
                    source=row[4],
                    signer_id=row[5],
                    frames=frames,
                )
        return None

    def get_all_variants(self, gloss: str) -> list[SignPoseSequence]:
        """Retrieve all variants of a sign.

        Args:
            gloss: The ASL gloss to look up.

        Returns:
            List of all variants for this sign.
        """
        variants = []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT gloss, variant_id, fps, duration_ms, source, signer_id, frames_json
                   FROM signs WHERE gloss = ? ORDER BY variant_id""",
                (gloss.upper(),),
            ).fetchall()

            for row in rows:
                frames = [PoseKeypoints.from_dict(f) for f in json.loads(row[6])]
                variants.append(SignPoseSequence(
                    gloss=row[0],
                    variant_id=row[1],
                    fps=row[2],
                    duration_ms=row[3],
                    source=row[4],
                    signer_id=row[5],
                    frames=frames,
                ))
        return variants

    def find_best_variant(
        self,
        gloss: str,
        previous_end_pose: PoseKeypoints | None = None,
    ) -> SignPoseSequence | None:
        """Find the best variant for smooth motion transition.

        Uses economy-of-motion principle: select the variant whose
        starting pose is closest to the previous sign's ending pose.

        Args:
            gloss: The ASL gloss to look up.
            previous_end_pose: The ending pose of the previous sign.

        Returns:
            Best matching SignPoseSequence, or None if not found.
        """
        variants = self.get_all_variants(gloss)
        if not variants:
            return None

        if previous_end_pose is None or len(variants) == 1:
            return variants[0]

        # Find variant with minimum transition distance
        best_variant = None
        min_distance = float("inf")

        for variant in variants:
            if variant.first_frame:
                distance = previous_end_pose.distance_to(variant.first_frame)
                if distance < min_distance:
                    min_distance = distance
                    best_variant = variant

        return best_variant or variants[0]

    def has_sign(self, gloss: str) -> bool:
        """Check if a sign exists in the dictionary.

        Args:
            gloss: The ASL gloss to check.

        Returns:
            True if sign exists, False otherwise.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM signs WHERE gloss = ? LIMIT 1",
                (gloss.upper(),),
            ).fetchone()
            return row is not None

    def list_glosses(self) -> list[str]:
        """List all unique glosses in the dictionary.

        Returns:
            List of all gloss strings.
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT gloss FROM signs ORDER BY gloss"
            ).fetchall()
            return [row[0] for row in rows]

    def count_signs(self) -> dict[str, int]:
        """Get counts of signs by source.

        Returns:
            Dictionary mapping source to count.
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) FROM signs GROUP BY source"
            ).fetchall()
            return {row[0]: row[1] for row in rows}

    def iter_signs(self, source: str | None = None) -> Iterator[SignPoseSequence]:
        """Iterate over all signs in the dictionary.

        Args:
            source: Optional filter by source dataset.

        Yields:
            SignPoseSequence objects.
        """
        with sqlite3.connect(self.db_path) as conn:
            if source:
                query = (
                    "SELECT gloss, variant_id, fps, duration_ms, source, signer_id, "
                    "frames_json FROM signs WHERE source = ?"
                )
                rows = conn.execute(query, (source,))
            else:
                query = (
                    "SELECT gloss, variant_id, fps, duration_ms, source, signer_id, "
                    "frames_json FROM signs"
                )
                rows = conn.execute(query)

            for row in rows:
                frames = [PoseKeypoints.from_dict(f) for f in json.loads(row[6])]
                yield SignPoseSequence(
                    gloss=row[0],
                    variant_id=row[1],
                    fps=row[2],
                    duration_ms=row[3],
                    source=row[4],
                    signer_id=row[5],
                    frames=frames,
                )


def interpolate_poses(
    start: PoseKeypoints,
    end: PoseKeypoints,
    num_frames: int,
) -> list[PoseKeypoints]:
    """Interpolate between two poses for smooth transitions.

    Uses linear interpolation (LERP) between keypoints.
    Future improvement: use quaternion SLERP for joint rotations.

    Args:
        start: Starting pose.
        end: Ending pose.
        num_frames: Number of interpolation frames to generate.

    Returns:
        List of interpolated PoseKeypoints.
    """
    if num_frames <= 0:
        return []

    frames = []
    for i in range(num_frames):
        t = (i + 1) / (num_frames + 1)  # Exclude endpoints

        # Linear interpolation
        body = start.body + t * (end.body - start.body)
        left_hand = start.left_hand + t * (end.left_hand - start.left_hand)
        right_hand = start.right_hand + t * (end.right_hand - start.right_hand)

        face = None
        if start.face is not None and end.face is not None:
            face = start.face + t * (end.face - start.face)

        frames.append(PoseKeypoints(
            body=body,
            left_hand=left_hand,
            right_hand=right_hand,
            face=face,
        ))

    return frames


def create_fingerspelling_sequence(
    letter: str,
    dictionary: PoseDictionary,
    duration_ms: int = 400,
) -> SignPoseSequence | None:
    """Create pose sequence for a single fingerspelled letter.

    Args:
        letter: Single letter to fingerspell.
        dictionary: Pose dictionary to look up letter.
        duration_ms: Duration for the letter pose.

    Returns:
        SignPoseSequence for the letter, or None if not found.
    """
    # Letters are stored as single-char glosses
    letter_gloss = letter.upper()
    sequence = dictionary.get_sign(letter_gloss)

    if sequence:
        return sequence

    # If not in dictionary, return None (could generate placeholder)
    return None


def generate_placeholder_keypoints(num_body: int = 33) -> PoseKeypoints:
    """Generate neutral/rest pose keypoints.

    Creates a basic standing pose with hands at sides.
    Used as fallback when sign is not in dictionary.

    Args:
        num_body: Number of body keypoints (33 for MediaPipe, 25 for OpenPose).

    Returns:
        PoseKeypoints in neutral pose.
    """
    # Normalized coordinates (0-1 range)
    # Body in center, standing position
    body = np.zeros((num_body, 3))
    body[:, 2] = 1.0  # Confidence

    # Simple body layout (approximate MediaPipe positions)
    # Head/face region
    body[0] = [0.5, 0.15, 1.0]  # Nose
    body[1:5] = [[0.48, 0.12, 1.0], [0.52, 0.12, 1.0],
                 [0.45, 0.12, 1.0], [0.55, 0.12, 1.0]]  # Eyes, ears

    # Shoulders, elbows, wrists
    body[11] = [0.4, 0.3, 1.0]   # Left shoulder
    body[12] = [0.6, 0.3, 1.0]   # Right shoulder
    body[13] = [0.35, 0.45, 1.0]  # Left elbow
    body[14] = [0.65, 0.45, 1.0]  # Right elbow
    body[15] = [0.32, 0.58, 1.0]  # Left wrist
    body[16] = [0.68, 0.58, 1.0]  # Right wrist

    # Hands at rest position (21 points each)
    left_hand = np.zeros((21, 3))
    left_hand[:, 0] = np.linspace(0.30, 0.35, 21)  # X spread
    left_hand[:, 1] = np.linspace(0.60, 0.70, 21)  # Y spread
    left_hand[:, 2] = 1.0  # Confidence

    right_hand = np.zeros((21, 3))
    right_hand[:, 0] = np.linspace(0.65, 0.70, 21)
    right_hand[:, 1] = np.linspace(0.60, 0.70, 21)
    right_hand[:, 2] = 1.0

    return PoseKeypoints(
        body=body,
        left_hand=left_hand,
        right_hand=right_hand,
        face=None,
    )
