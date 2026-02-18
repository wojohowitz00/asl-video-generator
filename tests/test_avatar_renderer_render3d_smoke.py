"""Smoke coverage for optional render3d dependencies."""

import json

import numpy as np
import pytest


def test_pyrender_optional_deps_probe_true_when_installed():
    """When render3d extras are installed, dependency probe should pass."""
    pytest.importorskip("pyrender")
    pytest.importorskip("trimesh")

    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    renderer = AvatarRenderer(RenderConfig(avatar_style="mesh", mesh_backend="pyrender"))
    assert renderer._is_pyrender_available() is True


def test_pyrender_offscreen_render_executes_or_skips_for_missing_gl_context():
    """Offscreen render should execute when possible, otherwise skip on missing GL context."""
    pytest.importorskip("pyrender")
    pytest.importorskip("trimesh")

    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    renderer = AvatarRenderer(
        RenderConfig(width=64, height=64, avatar_style="mesh", mesh_backend="pyrender")
    )
    vertices = np.array(
        [
            [-0.4, -0.4, 0.0],
            [0.4, -0.4, 0.0],
            [0.0, 0.45, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    try:
        rgb = renderer._render_with_pyrender(vertices, faces, {"translation": [0.0, 0.0, 0.0]})
    except Exception as exc:
        message = str(exc).lower()
        context_markers = [
            "display",
            "context",
            "opengl",
            "egl",
            "glx",
            "osmesa",
            "cannot connect",
            "no such display",
            "failed to create",
        ]
        if any(marker in message for marker in context_markers):
            pytest.skip(f"Offscreen GL context unavailable: {exc}")
        raise

    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8


def test_render_mesh_end_to_end_generates_output_with_pyrender_backend(tmp_path):
    """render_mesh should generate output under pyrender backend (or fallback safely)."""
    pytest.importorskip("pyrender")
    pytest.importorskip("trimesh")

    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    motion = {
        "fps": 12,
        "frames": [
            {
                "timestamp_ms": 0,
                "vertices": [[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.0, 0.45, 0.0]],
                "faces": [[0, 1, 2]],
                "translation": [0.0, 0.0, 0.0],
            },
            {
                "timestamp_ms": 80,
                "vertices": [[-0.38, -0.4, 0.0], [0.42, -0.4, 0.0], [0.02, 0.45, 0.0]],
                "faces": [[0, 1, 2]],
                "translation": [0.0, 0.0, 0.02],
            },
        ],
    }

    motion_path = tmp_path / "motion.json"
    output_path = tmp_path / "mesh.gif"
    motion_path.write_text(json.dumps(motion))

    renderer = AvatarRenderer(
        RenderConfig(
            width=64,
            height=64,
            fps=12,
            output_format="gif",
            avatar_style="mesh",
            mesh_backend="pyrender",
        )
    )

    result = renderer.render_mesh(motion_path, output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_pyrender_render_mesh_output_changes_with_camera_angle(tmp_path):
    """pyrender path should produce different artifacts for different camera angles."""
    pytest.importorskip("pyrender")
    pytest.importorskip("trimesh")

    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    motion = {
        "fps": 12,
        "frames": [
            {
                "timestamp_ms": 0,
                "vertices": [[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.0, 0.45, 0.0]],
                "faces": [[0, 1, 2]],
                "translation": [0.0, 0.0, 0.0],
            },
            {
                "timestamp_ms": 80,
                "vertices": [[-0.36, -0.4, 0.0], [0.44, -0.4, 0.0], [0.02, 0.45, 0.02]],
                "faces": [[0, 1, 2]],
                "translation": [0.0, 0.0, 0.02],
            },
        ],
    }

    motion_path = tmp_path / "motion_cam.json"
    out_a = tmp_path / "mesh_cam_a.gif"
    out_b = tmp_path / "mesh_cam_b.gif"
    motion_path.write_text(json.dumps(motion))

    renderer_a = AvatarRenderer(
        RenderConfig(
            width=64,
            height=64,
            fps=12,
            output_format="gif",
            avatar_style="mesh",
            mesh_backend="pyrender",
            camera_angle=(0.0, 0.0),
        )
    )
    renderer_b = AvatarRenderer(
        RenderConfig(
            width=64,
            height=64,
            fps=12,
            output_format="gif",
            avatar_style="mesh",
            mesh_backend="pyrender",
            camera_angle=(40.0, 20.0),
        )
    )

    result_a = renderer_a.render_mesh(motion_path, out_a)
    result_b = renderer_b.render_mesh(motion_path, out_b)

    telemetry_a = renderer_a.get_mesh_backend_telemetry()
    telemetry_b = renderer_b.get_mesh_backend_telemetry()
    if telemetry_a["counts"]["pyrender"] == 0 or telemetry_b["counts"]["pyrender"] == 0:
        pytest.skip("Native pyrender path unavailable at runtime; fallback path exercised instead.")

    assert result_a == out_a
    assert result_b == out_b
    assert out_a.exists()
    assert out_b.exists()
    assert out_a.read_bytes() != out_b.read_bytes()
