"""Tests for pose dictionary module."""

import numpy as np


def test_pose_keypoints_creation():
    """Test PoseKeypoints dataclass creation."""
    from asl_video_generator.pose_dictionary import PoseKeypoints

    kp = PoseKeypoints(
        body=np.zeros((33, 3)),
        left_hand=np.zeros((21, 3)),
        right_hand=np.zeros((21, 3)),
    )

    assert kp.body.shape == (33, 3)
    assert kp.left_hand.shape == (21, 3)
    assert kp.right_hand.shape == (21, 3)
    assert kp.face is None


def test_pose_keypoints_to_dict():
    """Test PoseKeypoints serialization."""
    from asl_video_generator.pose_dictionary import PoseKeypoints

    kp = PoseKeypoints(
        body=np.ones((33, 3)),
        left_hand=np.ones((21, 3)) * 2,
        right_hand=np.ones((21, 3)) * 3,
    )

    data = kp.to_dict()

    assert len(data["body"]) == 33
    assert len(data["left_hand"]) == 21
    assert len(data["right_hand"]) == 21
    assert data["face"] is None


def test_pose_keypoints_from_dict():
    """Test PoseKeypoints deserialization."""
    from asl_video_generator.pose_dictionary import PoseKeypoints

    data = {
        "body": [[1, 2, 3]] * 33,
        "left_hand": [[4, 5, 6]] * 21,
        "right_hand": [[7, 8, 9]] * 21,
        "face": None,
    }

    kp = PoseKeypoints.from_dict(data)

    assert kp.body.shape == (33, 3)
    assert kp.left_hand[0].tolist() == [4, 5, 6]


def test_pose_keypoints_distance():
    """Test distance calculation between poses."""
    from asl_video_generator.pose_dictionary import PoseKeypoints

    kp1 = PoseKeypoints(
        body=np.zeros((33, 3)),
        left_hand=np.zeros((21, 3)),
        right_hand=np.zeros((21, 3)),
    )

    kp2 = PoseKeypoints(
        body=np.ones((33, 3)),
        left_hand=np.ones((21, 3)),
        right_hand=np.ones((21, 3)),
    )

    distance = kp1.distance_to(kp2)

    assert distance > 0


def test_sign_pose_sequence():
    """Test SignPoseSequence creation."""
    from asl_video_generator.pose_dictionary import PoseKeypoints, SignPoseSequence

    kp = PoseKeypoints(
        body=np.zeros((33, 3)),
        left_hand=np.zeros((21, 3)),
        right_hand=np.zeros((21, 3)),
    )

    seq = SignPoseSequence(
        gloss="HELLO",
        frames=[kp, kp],
        fps=30,
    )

    assert seq.gloss == "HELLO"
    assert len(seq.frames) == 2
    assert seq.duration_ms > 0


def test_pose_dictionary_add_get(tmp_path):
    """Test PoseDictionary add and retrieve."""
    from asl_video_generator.pose_dictionary import (
        PoseDictionary,
        PoseKeypoints,
        SignPoseSequence,
    )

    db_path = tmp_path / "test.db"
    dictionary = PoseDictionary(db_path=db_path)

    kp = PoseKeypoints(
        body=np.ones((33, 3)),
        left_hand=np.ones((21, 3)),
        right_hand=np.ones((21, 3)),
    )

    seq = SignPoseSequence(
        gloss="TEST",
        frames=[kp],
        fps=30,
        source="test",
    )

    # Add
    dictionary.add_sign(seq)

    # Get
    retrieved = dictionary.get_sign("TEST")

    assert retrieved is not None
    assert retrieved.gloss == "TEST"
    assert len(retrieved.frames) == 1


def test_pose_dictionary_has_sign(tmp_path):
    """Test PoseDictionary has_sign check."""
    from asl_video_generator.pose_dictionary import (
        PoseDictionary,
        PoseKeypoints,
        SignPoseSequence,
    )

    db_path = tmp_path / "test.db"
    dictionary = PoseDictionary(db_path=db_path)

    kp = PoseKeypoints(
        body=np.ones((33, 3)),
        left_hand=np.ones((21, 3)),
        right_hand=np.ones((21, 3)),
    )

    seq = SignPoseSequence(gloss="EXISTS", frames=[kp], fps=30, source="test")
    dictionary.add_sign(seq)

    assert dictionary.has_sign("EXISTS")
    assert dictionary.has_sign("exists")  # Case insensitive
    assert not dictionary.has_sign("NOTEXISTS")


def test_interpolate_poses():
    """Test pose interpolation."""
    from asl_video_generator.pose_dictionary import PoseKeypoints, interpolate_poses

    start = PoseKeypoints(
        body=np.zeros((33, 3)),
        left_hand=np.zeros((21, 3)),
        right_hand=np.zeros((21, 3)),
    )

    end = PoseKeypoints(
        body=np.ones((33, 3)),
        left_hand=np.ones((21, 3)),
        right_hand=np.ones((21, 3)),
    )

    interpolated = interpolate_poses(start, end, num_frames=5)

    assert len(interpolated) == 5
    # Middle frame should be roughly in between
    mid = interpolated[2]
    assert 0.3 < mid.body[0, 0] < 0.7


def test_generate_placeholder_keypoints():
    """Test placeholder keypoint generation."""
    from asl_video_generator.pose_dictionary import generate_placeholder_keypoints

    kp = generate_placeholder_keypoints()

    assert kp.body.shape == (33, 3)
    assert kp.left_hand.shape == (21, 3)
    assert kp.right_hand.shape == (21, 3)
    # Confidence should be 1.0
    assert kp.body[:, 2].mean() == 1.0
