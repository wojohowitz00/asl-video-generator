"""Tests for vocabulary motion generation."""

import json

import pytest


def test_generate_requires_model_path_when_not_placeholder():
    """Non-placeholder mode should fail fast without model path."""
    from asl_video_generator.vocabulary_generator import VocabularyGenerator

    generator = VocabularyGenerator(use_placeholder=False)

    with pytest.raises(ValueError, match="model_path"):
        generator.generate("hello")


def test_generate_from_lite_checkpoint(tmp_path):
    """Generator should produce model-backed motion from lite checkpoint."""
    from asl_video_generator.vocabulary_generator import VocabularyGenerator

    checkpoint = {
        "model_type": "wsigngen-lite",
        "fps": 24,
        "num_frames": 12,
        "mesh_format": "smplh",
        "body_pose_dim": 72,
        "hand_pose_dim": 45,
        "expression_dim": 10,
        "amplitude": 0.35,
        "translation_scale": 0.03,
        "seed": 17,
    }

    model_path = tmp_path / "wsigngen_lite.json"
    model_path.write_text(json.dumps(checkpoint))

    generator = VocabularyGenerator(model_path=model_path, use_placeholder=False)
    motion = generator.generate("hello")

    assert motion.word == "hello"
    assert motion.asl_gloss == "HELLO"
    assert motion.fps == 24
    assert motion.mesh_format == "smplh"
    assert len(motion.frames) == 12
    assert motion.total_duration_ms == 500

    first = motion.frames[0]
    assert len(first.body_pose) == 72
    assert len(first.left_hand_pose) == 45
    assert len(first.right_hand_pose) == 45
    assert len(first.expression or []) == 10

    # Ensure this is not the placeholder zero-left-hand path.
    assert any(abs(v) > 1e-6 for v in first.left_hand_pose)
