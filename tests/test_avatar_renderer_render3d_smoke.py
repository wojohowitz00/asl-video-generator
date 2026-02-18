"""Smoke coverage for optional render3d dependencies."""

import pytest


def test_pyrender_optional_deps_probe_true_when_installed():
    """When render3d extras are installed, dependency probe should pass."""
    pytest.importorskip("pyrender")
    pytest.importorskip("trimesh")

    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    renderer = AvatarRenderer(RenderConfig(avatar_style="mesh", mesh_backend="pyrender"))
    assert renderer._is_pyrender_available() is True
