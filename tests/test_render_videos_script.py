"""Tests for render_videos script mesh-backend controls."""

import importlib.util
from pathlib import Path


def _load_render_videos_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "render_videos.py"
    spec = importlib.util.spec_from_file_location("render_videos_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_parser_exposes_mesh_backend_controls():
    """Script parser should expose avatar/mesh backend and camera controls."""
    module = _load_render_videos_module()

    parser = module.create_parser()
    args = parser.parse_args([])

    assert args.avatar_style == "skeleton"
    assert args.mesh_backend == "software_3d"
    assert args.camera_azimuth == 0.0
    assert args.camera_elevation == 0.0


def test_build_render_config_maps_mesh_backend_controls():
    """Parsed mesh control args should map into RenderConfig values."""
    module = _load_render_videos_module()

    parser = module.create_parser()
    args = parser.parse_args(
        [
            "--avatar-style",
            "mesh",
            "--mesh-backend",
            "pyrender",
            "--camera-azimuth",
            "35",
            "--camera-elevation",
            "15",
            "--fps",
            "24",
            "--width",
            "640",
            "--height",
            "360",
        ]
    )

    config = module.build_render_config(args)

    assert config.avatar_style == "mesh"
    assert config.mesh_backend == "pyrender"
    assert config.camera_angle == (35.0, 15.0)
    assert config.fps == 24
    assert config.width == 640
    assert config.height == 360
