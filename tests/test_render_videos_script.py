"""Tests for render_videos script mesh-backend controls."""

import importlib.util
import json
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


def test_main_routes_mesh_style_to_render_mesh(monkeypatch, tmp_path):
    """main should route mesh avatar style to AvatarRenderer.render_mesh."""
    module = _load_render_videos_module()

    input_dir = tmp_path / "poses"
    output_dir = tmp_path / "videos"
    input_dir.mkdir()
    motion_path = input_dir / "sample.json"
    motion_path.write_text(json.dumps({"frames": []}))

    calls = {"poses": 0, "mesh": 0}

    class _FakeRenderer:
        def __init__(self, _config):
            pass

        def render_poses(self, _input_path, output_path):
            calls["poses"] += 1
            output_path.write_text("poses")
            return output_path

        def render_mesh(self, _input_path, output_path):
            calls["mesh"] += 1
            output_path.write_text("mesh")
            return output_path

    monkeypatch.setattr(module, "AvatarRenderer", _FakeRenderer)

    module.main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--format",
            "gif",
            "--avatar-style",
            "mesh",
        ]
    )

    assert calls["mesh"] == 1
    assert calls["poses"] == 0


def test_main_routes_skeleton_style_to_render_poses(monkeypatch, tmp_path):
    """main should route non-mesh avatar styles to AvatarRenderer.render_poses."""
    module = _load_render_videos_module()

    input_dir = tmp_path / "poses"
    output_dir = tmp_path / "videos"
    input_dir.mkdir()
    pose_path = input_dir / "sample.json"
    pose_path.write_text(json.dumps({"frames": []}))

    calls = {"poses": 0, "mesh": 0}

    class _FakeRenderer:
        def __init__(self, _config):
            pass

        def render_poses(self, _input_path, output_path):
            calls["poses"] += 1
            output_path.write_text("poses")
            return output_path

        def render_mesh(self, _input_path, output_path):
            calls["mesh"] += 1
            output_path.write_text("mesh")
            return output_path

    monkeypatch.setattr(module, "AvatarRenderer", _FakeRenderer)

    module.main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--format",
            "gif",
            "--avatar-style",
            "skeleton",
        ]
    )

    assert calls["poses"] == 1
    assert calls["mesh"] == 0


def test_main_routes_stylized_style_to_render_mesh(monkeypatch, tmp_path):
    """main should route stylized avatar style to AvatarRenderer.render_mesh."""
    module = _load_render_videos_module()

    input_dir = tmp_path / "poses"
    output_dir = tmp_path / "videos"
    input_dir.mkdir()
    motion_path = input_dir / "sample.json"
    motion_path.write_text(json.dumps({"frames": []}))

    calls = {"poses": 0, "mesh": 0}

    class _FakeRenderer:
        def __init__(self, _config):
            pass

        def render_poses(self, _input_path, output_path):
            calls["poses"] += 1
            output_path.write_text("poses")
            return output_path

        def render_mesh(self, _input_path, output_path):
            calls["mesh"] += 1
            output_path.write_text("mesh")
            return output_path

    monkeypatch.setattr(module, "AvatarRenderer", _FakeRenderer)

    module.main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--format",
            "gif",
            "--avatar-style",
            "stylized",
        ]
    )

    assert calls["mesh"] == 1
    assert calls["poses"] == 0
