"""Smoke coverage for optional render3d dependencies."""

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
