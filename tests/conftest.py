"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_gloss_sequence():
    """Create a sample GlossSequence for testing."""
    from asl_video_generator.gloss_translator import GlossSequence, NonManualMarkers

    return GlossSequence(
        english="Hello, how are you?",
        gloss=["HELLO", "HOW", "YOU"],
        nmm=NonManualMarkers(
            is_question=True,
            question_type="wh",
            eyebrow_position="furrowed",
        ),
        estimated_duration_ms=1500,
    )


@pytest.fixture
def sample_pose_sequence():
    """Create a sample PoseSequence for testing."""
    from asl_video_generator.pose_generator import PoseFrame, PoseSequence

    frames = [
        PoseFrame(
            timestamp_ms=i * 33,
            body_keypoints=[(0.5, 0.5, 1.0)] * 25,
            left_hand=[(0.3, 0.6, 1.0)] * 21,
            right_hand=[(0.7, 0.6, 1.0)] * 21,
        )
        for i in range(30)
    ]

    return PoseSequence(
        english="Test",
        gloss=["TEST"],
        frames=frames,
        fps=30,
        total_duration_ms=1000,
    )


@pytest.fixture
def temp_pose_dictionary(tmp_path):
    """Create a temporary pose dictionary with sample data."""
    import numpy as np
    from asl_video_generator.pose_dictionary import (
        PoseDictionary,
        PoseKeypoints,
        SignPoseSequence,
    )

    db_path = tmp_path / "test_poses.db"
    dictionary = PoseDictionary(db_path=db_path)

    # Add some sample signs
    for gloss in ["HELLO", "HOW", "YOU", "THANK-YOU"]:
        kp = PoseKeypoints(
            body=np.random.rand(33, 3),
            left_hand=np.random.rand(21, 3),
            right_hand=np.random.rand(21, 3),
        )
        kp.body[:, 2] = 1.0
        kp.left_hand[:, 2] = 1.0
        kp.right_hand[:, 2] = 1.0

        seq = SignPoseSequence(
            gloss=gloss,
            frames=[kp] * 10,
            fps=30,
            source="test",
        )
        dictionary.add_sign(seq)

    return dictionary
