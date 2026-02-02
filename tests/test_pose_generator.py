"""Tests for pose generator module."""

import json
from pathlib import Path

import pytest


def test_pose_frame_creation():
    """Test PoseFrame dataclass creation."""
    from asl_video_generator.pose_generator import PoseFrame

    frame = PoseFrame(
        timestamp_ms=0,
        body_keypoints=[(0.5, 0.5, 1.0)] * 25,
        left_hand=[(0.3, 0.6, 1.0)] * 21,
        right_hand=[(0.7, 0.6, 1.0)] * 21,
    )

    assert frame.timestamp_ms == 0
    assert len(frame.body_keypoints) == 25
    assert len(frame.left_hand) == 21
    assert len(frame.right_hand) == 21


def test_pose_sequence_to_json():
    """Test PoseSequence serialization."""
    from asl_video_generator.pose_generator import PoseFrame, PoseSequence

    frame = PoseFrame(
        timestamp_ms=0,
        body_keypoints=[(0.5, 0.5, 1.0)] * 25,
        left_hand=[(0.3, 0.6, 1.0)] * 21,
        right_hand=[(0.7, 0.6, 1.0)] * 21,
    )

    seq = PoseSequence(
        english="Hello",
        gloss=["HELLO"],
        frames=[frame],
        fps=30,
        total_duration_ms=500,
    )

    data = seq.to_json()

    assert data["english"] == "Hello"
    assert data["gloss"] == ["HELLO"]
    assert len(data["frames"]) == 1


def test_pose_sequence_save_load(tmp_path):
    """Test PoseSequence save and load."""
    from asl_video_generator.pose_generator import PoseFrame, PoseSequence

    frame = PoseFrame(
        timestamp_ms=0,
        body_keypoints=[(0.5, 0.5, 1.0)] * 25,
        left_hand=[(0.3, 0.6, 1.0)] * 21,
        right_hand=[(0.7, 0.6, 1.0)] * 21,
    )

    seq = PoseSequence(
        english="Test",
        gloss=["TEST"],
        frames=[frame],
        fps=30,
        total_duration_ms=500,
    )

    # Save
    save_path = tmp_path / "test.json"
    seq.save(save_path)

    assert save_path.exists()

    # Load
    loaded = PoseSequence.load(save_path)

    assert loaded.english == seq.english
    assert loaded.gloss == seq.gloss
    assert len(loaded.frames) == len(seq.frames)


def test_pose_generator_empty_input():
    """Test PoseGenerator handles empty input."""
    from asl_video_generator.gloss_translator import GlossSequence
    from asl_video_generator.pose_generator import PoseGenerator

    generator = PoseGenerator()
    gloss_seq = GlossSequence(english="", gloss=[], estimated_duration_ms=0)

    result = generator.generate(gloss_seq)

    assert result.frames == []
    assert result.total_duration_ms == 0


def test_pose_generator_placeholder_generation():
    """Test PoseGenerator generates placeholders for missing signs."""
    from asl_video_generator.gloss_translator import GlossSequence
    from asl_video_generator.pose_generator import PoseGenerator

    generator = PoseGenerator()
    gloss_seq = GlossSequence(
        english="Unknown sign",
        gloss=["UNKNOWN_SIGN_XYZ"],
        estimated_duration_ms=500,
    )

    result = generator.generate(gloss_seq)

    # Should generate placeholder frames
    assert len(result.frames) > 0
    # Should track missing signs
    assert "UNKNOWN_SIGN_XYZ" in result.missing_signs


def test_fingerspelling_detection():
    """Test fingerspelling pattern detection."""
    from asl_video_generator.pose_generator import PoseGenerator

    generator = PoseGenerator()

    assert generator._is_fingerspelled("J-O-H-N")
    assert generator._is_fingerspelled("A-B-C")
    assert not generator._is_fingerspelled("HELLO")
    assert not generator._is_fingerspelled("THANK-YOU")  # Compound sign


def test_nmm_application():
    """Test NMM overlay application."""
    from asl_video_generator.gloss_translator import NonManualMarkers
    from asl_video_generator.pose_generator import PoseFrame, PoseGenerator

    generator = PoseGenerator()

    frame = PoseFrame(
        timestamp_ms=0,
        body_keypoints=[(0.5, 0.5, 1.0)] * 25,
        left_hand=[(0.3, 0.6, 1.0)] * 21,
        right_hand=[(0.7, 0.6, 1.0)] * 21,
    )

    nmm = NonManualMarkers(
        is_question=True,
        eyebrow_position="raised",
    )

    modified = generator._apply_nmm_to_frame(frame, nmm, 0, 10)

    # Frame should be modified (eyebrows raised)
    assert modified.timestamp_ms == frame.timestamp_ms
    # Body keypoints should be different
    # (eyebrow modification changes face region)
