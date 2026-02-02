"""Dictionary-based pose generator for ASL sentence animation.

This module generates skeletal poses from ASL gloss sequences using:
1. Dictionary lookup for pre-recorded sign poses
2. Motion matching for smooth transitions
3. NMM overlay for grammatical markers
4. Interpolation for seamless blending between signs
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .config import PipelineConfig, load_config_from_env
from .gloss_translator import GlossSequence, NonManualMarkers
from .pose_dictionary import (
    PoseDictionary,
    PoseKeypoints,
    SignPoseSequence,
    generate_placeholder_keypoints,
    interpolate_poses,
)


@dataclass
class PoseFrame:
    """A single frame of skeletal pose data."""

    timestamp_ms: int
    # Upper body keypoints (33 MediaPipe or 25 OpenPose format)
    body_keypoints: list[tuple[float, float, float]]  # (x, y, confidence)
    # Left hand keypoints (21 points)
    left_hand: list[tuple[float, float, float]]
    # Right hand keypoints (21 points)
    right_hand: list[tuple[float, float, float]]
    # Face keypoints (468 MediaPipe or 70 OpenPose) - optional
    face: list[tuple[float, float, float]] | None = None

    @classmethod
    def from_keypoints(cls, kp: PoseKeypoints, timestamp_ms: int) -> "PoseFrame":
        """Create PoseFrame from PoseKeypoints."""
        return cls(
            timestamp_ms=timestamp_ms,
            body_keypoints=[tuple(p) for p in kp.body.tolist()],
            left_hand=[tuple(p) for p in kp.left_hand.tolist()],
            right_hand=[tuple(p) for p in kp.right_hand.tolist()],
            face=[tuple(p) for p in kp.face.tolist()] if kp.face is not None else None,
        )


@dataclass
class PoseSequence:
    """A sequence of pose frames for an ASL phrase."""

    english: str
    gloss: list[str]
    frames: list[PoseFrame]
    fps: int = 30
    total_duration_ms: int = 0
    difficulty: Literal["beginner", "intermediate", "advanced"] = "beginner"
    # Track which signs are missing from dictionary
    missing_signs: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        """Convert to JSON-serializable format."""
        return {
            "english": self.english,
            "gloss": self.gloss,
            "fps": self.fps,
            "total_duration_ms": self.total_duration_ms,
            "difficulty": self.difficulty,
            "missing_signs": self.missing_signs,
            "frames": [
                {
                    "timestamp_ms": f.timestamp_ms,
                    "body": f.body_keypoints,
                    "left_hand": f.left_hand,
                    "right_hand": f.right_hand,
                    "face": f.face,
                }
                for f in self.frames
            ],
        }

    def save(self, path: Path) -> None:
        """Save pose sequence to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PoseSequence":
        """Load pose sequence from JSON file."""
        data = json.loads(path.read_text())
        frames = [
            PoseFrame(
                timestamp_ms=f["timestamp_ms"],
                body_keypoints=[tuple(p) for p in f["body"]],
                left_hand=[tuple(p) for p in f["left_hand"]],
                right_hand=[tuple(p) for p in f["right_hand"]],
                face=[tuple(p) for p in f["face"]] if f.get("face") else None,
            )
            for f in data["frames"]
        ]
        return cls(
            english=data["english"],
            gloss=data["gloss"],
            frames=frames,
            fps=data["fps"],
            total_duration_ms=data["total_duration_ms"],
            difficulty=data.get("difficulty", "beginner"),
            missing_signs=data.get("missing_signs", []),
        )


class PoseGenerator:
    """Generate ASL skeletal poses from gloss sequences.

    Uses a dictionary-based approach:
    1. Look up each sign in the pose dictionary
    2. Select optimal variants using motion matching
    3. Apply NMM overlays for grammatical markers
    4. Interpolate between signs for smooth transitions
    """

    def __init__(
        self,
        dictionary: PoseDictionary | None = None,
        config: PipelineConfig | None = None,
        blend_frames: int | None = None,
    ):
        """Initialize the pose generator.

        Args:
            dictionary: Pose dictionary for sign lookup. Creates new if None.
            config: Pipeline configuration.
            blend_frames: Number of frames for interpolation between signs.
        """
        self.config = config or load_config_from_env()
        self.dictionary = dictionary or PoseDictionary(config=self.config)
        self.blend_frames = blend_frames or self.config.blend_frames
        self.fps = self.config.pose_fps

    def generate(
        self,
        gloss_sequence: GlossSequence,
    ) -> PoseSequence:
        """Generate pose sequence from gloss sequence.

        Args:
            gloss_sequence: GlossSequence with gloss array and NMM annotations.

        Returns:
            PoseSequence with skeletal animation data.
        """
        if not gloss_sequence.gloss:
            return PoseSequence(
                english=gloss_sequence.english,
                gloss=[],
                frames=[],
                fps=self.fps,
                total_duration_ms=0,
            )

        # Step 1: Look up signs and select optimal variants
        sign_sequences = self._lookup_signs(gloss_sequence.gloss)

        # Step 2: Concatenate with interpolation
        frames, missing = self._concatenate_with_interpolation(sign_sequences)

        # Step 3: Apply NMM overlays
        frames = self._apply_nmm_overlays(frames, gloss_sequence.nmm_spans)

        # Calculate total duration
        total_duration = int(len(frames) / self.fps * 1000) if frames else 0

        return PoseSequence(
            english=gloss_sequence.english,
            gloss=gloss_sequence.gloss,
            frames=frames,
            fps=self.fps,
            total_duration_ms=total_duration,
            difficulty=gloss_sequence.difficulty,
            missing_signs=missing,
        )

    def generate_from_text(
        self,
        english: str,
        gloss: list[str] | None = None,
    ) -> PoseSequence:
        """Generate poses from English text (convenience method).

        Args:
            english: English text.
            gloss: Optional pre-computed gloss. If None, uses simple uppercase split.

        Returns:
            PoseSequence with skeletal animation.
        """
        if gloss is None:
            # Simple fallback - just uppercase and split
            gloss = english.upper().replace("?", "").replace(".", "").replace(",", "").split()

        gloss_seq = GlossSequence(
            english=english,
            gloss=gloss,
            estimated_duration_ms=len(gloss) * 500,
        )
        return self.generate(gloss_seq)

    def _lookup_signs(
        self,
        glosses: list[str],
    ) -> list[tuple[str, SignPoseSequence | None]]:
        """Look up signs in dictionary with motion matching.

        Args:
            glosses: List of ASL glosses.

        Returns:
            List of (gloss, SignPoseSequence or None) tuples.
        """
        results = []
        previous_end_pose: PoseKeypoints | None = None

        for gloss in glosses:
            # Check if fingerspelled (single letters with hyphens)
            if self._is_fingerspelled(gloss):
                sequence = self._generate_fingerspelling(gloss, previous_end_pose)
            else:
                # Look up in dictionary with motion matching
                sequence = self.dictionary.find_best_variant(gloss, previous_end_pose)

            results.append((gloss, sequence))

            # Update previous end pose for next lookup
            if sequence and sequence.last_frame:
                previous_end_pose = sequence.last_frame

        return results

    def _is_fingerspelled(self, gloss: str) -> bool:
        """Check if gloss represents fingerspelled word."""
        if "-" not in gloss:
            return False
        parts = gloss.split("-")
        return all(len(p) == 1 and p.isalpha() for p in parts)

    def _generate_fingerspelling(
        self,
        gloss: str,
        previous_end_pose: PoseKeypoints | None,
    ) -> SignPoseSequence | None:
        """Generate pose sequence for fingerspelled word.

        Args:
            gloss: Fingerspelled gloss like "J-O-H-N".
            previous_end_pose: Previous pose for motion matching.

        Returns:
            Combined SignPoseSequence for all letters.
        """
        letters = gloss.split("-")
        all_frames: list[PoseKeypoints] = []
        current_end_pose = previous_end_pose

        for letter in letters:
            letter_seq = self.dictionary.get_sign(letter)
            if letter_seq:
                # Add interpolation between letters
                if current_end_pose and letter_seq.first_frame:
                    blend = interpolate_poses(
                        current_end_pose,
                        letter_seq.first_frame,
                        num_frames=max(5, self.blend_frames // 2),  # Shorter blend for letters
                    )
                    all_frames.extend(blend)

                all_frames.extend(letter_seq.frames)
                current_end_pose = letter_seq.last_frame
            else:
                # Generate placeholder for missing letter
                placeholder = generate_placeholder_keypoints()
                # Hold for ~300ms
                hold_frames = int(0.3 * self.fps)
                all_frames.extend([placeholder] * hold_frames)
                current_end_pose = placeholder

        if not all_frames:
            return None

        return SignPoseSequence(
            gloss=gloss,
            frames=all_frames,
            fps=self.fps,
            source="fingerspelling",
        )

    def _concatenate_with_interpolation(
        self,
        sign_sequences: list[tuple[str, SignPoseSequence | None]],
    ) -> tuple[list[PoseFrame], list[str]]:
        """Concatenate sign sequences with smooth interpolation.

        Args:
            sign_sequences: List of (gloss, sequence) tuples.

        Returns:
            Tuple of (list of PoseFrames, list of missing signs).
        """
        frames: list[PoseFrame] = []
        missing_signs: list[str] = []
        current_time_ms = 0
        previous_end: PoseKeypoints | None = None

        for gloss, sequence in sign_sequences:
            if sequence is None:
                # Sign not in dictionary - generate placeholder
                missing_signs.append(gloss)
                placeholder_frames = self._generate_placeholder_sign(
                    gloss, current_time_ms, previous_end
                )
                frames.extend(placeholder_frames)
                if placeholder_frames:
                    current_time_ms = placeholder_frames[-1].timestamp_ms + int(1000 / self.fps)
                    # Extract last keypoints for next transition
                    last = placeholder_frames[-1]
                    previous_end = PoseKeypoints(
                        body=np.array(last.body_keypoints),
                        left_hand=np.array(last.left_hand),
                        right_hand=np.array(last.right_hand),
                        face=np.array(last.face) if last.face else None,
                    )
                continue

            # Add interpolation from previous sign
            if previous_end and sequence.first_frame:
                blend_keypoints = interpolate_poses(
                    previous_end,
                    sequence.first_frame,
                    num_frames=self.blend_frames,
                )
                for kp in blend_keypoints:
                    frames.append(PoseFrame.from_keypoints(kp, current_time_ms))
                    current_time_ms += int(1000 / self.fps)

            # Add sign frames
            for kp in sequence.frames:
                frames.append(PoseFrame.from_keypoints(kp, current_time_ms))
                current_time_ms += int(1000 / self.fps)

            previous_end = sequence.last_frame

        return frames, missing_signs

    def _generate_placeholder_sign(
        self,
        gloss: str,
        start_time_ms: int,
        previous_end: PoseKeypoints | None,
    ) -> list[PoseFrame]:
        """Generate placeholder frames for a missing sign.

        Creates a simple hand motion to indicate a sign is being made.

        Args:
            gloss: The missing gloss.
            start_time_ms: Start timestamp.
            previous_end: Previous pose for transition.

        Returns:
            List of placeholder PoseFrames.
        """
        frames: list[PoseFrame] = []
        duration_ms = 500  # Default duration for placeholder
        num_frames = int(duration_ms / 1000 * self.fps)

        # Create a simple "signing" motion
        base = generate_placeholder_keypoints()
        current_time = start_time_ms

        # If we have previous pose, interpolate to neutral first
        if previous_end:
            blend = interpolate_poses(previous_end, base, num_frames=self.blend_frames // 2)
            for kp in blend:
                frames.append(PoseFrame.from_keypoints(kp, current_time))
                current_time += int(1000 / self.fps)

        # Generate simple motion (hands move slightly)
        for i in range(num_frames):
            t = i / num_frames
            modified = self._apply_simple_motion(base, t)
            frames.append(PoseFrame.from_keypoints(modified, current_time))
            current_time += int(1000 / self.fps)

        return frames

    def _apply_simple_motion(self, base: PoseKeypoints, t: float) -> PoseKeypoints:
        """Apply simple motion to base pose for placeholder animation.

        Args:
            base: Base pose keypoints.
            t: Time parameter (0 to 1).

        Returns:
            Modified PoseKeypoints.
        """
        # Simple sinusoidal motion for hands
        offset = np.sin(t * np.pi * 2) * 0.05

        right_hand = base.right_hand.copy()
        right_hand[:, 1] += offset  # Move up/down

        left_hand = base.left_hand.copy()
        left_hand[:, 1] -= offset * 0.5  # Smaller opposite motion

        return PoseKeypoints(
            body=base.body.copy(),
            left_hand=left_hand,
            right_hand=right_hand,
            face=base.face.copy() if base.face is not None else None,
        )

    def _apply_nmm_overlays(
        self,
        frames: list[PoseFrame],
        nmm_spans: list[NonManualMarkers],
    ) -> list[PoseFrame]:
        """Apply Non-Manual Marker modifications to pose frames.

        NMMs modify the face and head keypoints to express:
        - Eyebrow position (raised for yes/no Q, furrowed for wh-Q)
        - Head movement (shake for negation, nod for affirmation)
        - Facial expressions

        Args:
            frames: List of PoseFrames to modify.
            nmm_spans: List of NMM annotations with span info.

        Returns:
            Modified list of PoseFrames.
        """
        if not frames or not nmm_spans:
            return frames

        # For now, apply the primary NMM to all frames
        # Future: use span indices for localized NMM application
        primary_nmm = nmm_spans[0] if nmm_spans else None
        if not primary_nmm:
            return frames

        modified_frames = []
        for i, frame in enumerate(frames):
            modified = self._apply_nmm_to_frame(frame, primary_nmm, i, len(frames))
            modified_frames.append(modified)

        return modified_frames

    def _apply_nmm_to_frame(
        self,
        frame: PoseFrame,
        nmm: NonManualMarkers,
        frame_idx: int,
        total_frames: int,
    ) -> PoseFrame:
        """Apply NMM modifications to a single frame.

        Args:
            frame: Original PoseFrame.
            nmm: Non-Manual Markers to apply.
            frame_idx: Current frame index.
            total_frames: Total number of frames.

        Returns:
            Modified PoseFrame.
        """
        # Copy the frame
        body = list(frame.body_keypoints)

        # MediaPipe face landmark indices (approximate)
        # Eyebrows: landmarks around indices 70-80 (left) and 300-310 (right)
        # For body pose, we'll modify the face region keypoints

        # Apply eyebrow modifications
        if nmm.eyebrow_position == "raised":
            # Move eyebrow-related keypoints up
            body = self._modify_eyebrows(body, offset=-0.02)  # Negative = up
        elif nmm.eyebrow_position == "furrowed":
            # Move eyebrow keypoints down and together
            body = self._modify_eyebrows(body, offset=0.01)

        # Apply head movement
        t = frame_idx / max(total_frames - 1, 1)
        if nmm.head_movement == "shake":
            # Oscillating head shake for negation
            shake_offset = np.sin(t * np.pi * 4) * 0.02  # 2 complete shakes
            body = self._modify_head_position(body, x_offset=shake_offset)
        elif nmm.head_movement == "nod":
            # Single nod
            nod_offset = np.sin(t * np.pi * 2) * 0.015
            body = self._modify_head_position(body, y_offset=nod_offset)
        elif nmm.head_movement == "forward":
            # Lean forward (for questions)
            body = self._modify_head_position(body, y_offset=0.01)

        return PoseFrame(
            timestamp_ms=frame.timestamp_ms,
            body_keypoints=body,
            left_hand=frame.left_hand,
            right_hand=frame.right_hand,
            face=frame.face,
        )

    def _modify_eyebrows(
        self,
        body: list[tuple[float, float, float]],
        offset: float,
    ) -> list[tuple[float, float, float]]:
        """Modify eyebrow keypoints.

        Args:
            body: Body keypoints list.
            offset: Y offset to apply (negative = up).

        Returns:
            Modified body keypoints.
        """
        # MediaPipe body pose indices for face region: 0-10
        # We'll modify the eye-related landmarks
        modified = list(body)
        for i in range(min(5, len(modified))):  # Face region
            x, y, c = modified[i]
            modified[i] = (x, y + offset, c)
        return modified

    def _modify_head_position(
        self,
        body: list[tuple[float, float, float]],
        x_offset: float = 0,
        y_offset: float = 0,
    ) -> list[tuple[float, float, float]]:
        """Modify head position keypoints.

        Args:
            body: Body keypoints list.
            x_offset: Horizontal offset.
            y_offset: Vertical offset.

        Returns:
            Modified body keypoints.
        """
        # Modify head/face region (roughly first 11 landmarks in MediaPipe)
        modified = list(body)
        for i in range(min(11, len(modified))):
            x, y, c = modified[i]
            modified[i] = (x + x_offset, y + y_offset, c)
        return modified


def generate_poses_batch(
    sentences: list[dict],
    output_dir: Path,
    config: PipelineConfig | None = None,
) -> list[Path]:
    """Generate pose sequences for a batch of sentences.

    Args:
        sentences: List of dicts with 'text', 'id', and optional 'gloss' keys.
        output_dir: Directory to save pose JSON files.
        config: Pipeline configuration.

    Returns:
        List of paths to generated pose files.
    """
    config = config or load_config_from_env()
    generator = PoseGenerator(config=config)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for sentence in sentences:
        gloss_seq = GlossSequence(
            english=sentence["text"],
            gloss=sentence.get("gloss", sentence["text"].upper().split()),
            estimated_duration_ms=len(sentence.get("gloss", [])) * 500,
        )

        pose = generator.generate(gloss_seq)
        output_path = output_dir / f"{sentence['id']}.json"
        pose.save(output_path)
        paths.append(output_path)
        print(f"Generated: {output_path}")

    return paths
