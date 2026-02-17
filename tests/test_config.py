"""Tests for configuration module."""

import os
from unittest.mock import patch


def test_device_detection():
    """Test device detection returns valid DeviceType."""
    from asl_video_generator.config import DeviceType, detect_device

    device = detect_device()
    assert device in (DeviceType.MPS, DeviceType.CUDA, DeviceType.CPU)


def test_quality_presets_exist():
    """Test all quality presets have settings."""
    from asl_video_generator.config import QUALITY_PRESETS, QualityPreset

    for preset in QualityPreset:
        assert preset in QUALITY_PRESETS
        settings = QUALITY_PRESETS[preset]
        assert settings.width > 0
        assert settings.height > 0
        assert settings.fps > 0


def test_pipeline_config_defaults():
    """Test PipelineConfig has sensible defaults."""
    from asl_video_generator.config import PipelineConfig

    config = PipelineConfig()

    assert config.use_fp16 is True
    assert config.enable_attention_slicing is True
    assert config.pose_fps == 30
    assert config.blend_frames > 0


def test_validate_mps_availability():
    """Test MPS validation returns valid structure."""
    from asl_video_generator.config import validate_mps_availability

    result = validate_mps_availability()

    assert "device" in result
    assert "available" in result
    assert isinstance(result["available"], bool)


def test_env_overrides():
    """Test environment variable overrides work."""
    from asl_video_generator.config import QualityPreset, load_config_from_env

    with patch.dict(os.environ, {"ASL_QUALITY": "preview"}):
        # Clear cache to force re-detection
        from asl_video_generator.config import detect_device
        detect_device.cache_clear()

        config = load_config_from_env()
        assert config.quality == QualityPreset.PREVIEW
