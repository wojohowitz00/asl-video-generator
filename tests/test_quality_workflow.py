"""Tests guarding expected CI workflow coverage."""

from pathlib import Path


def test_quality_workflow_includes_render3d_smoke_job():
    """Quality workflow should include render3d smoke coverage job."""
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "quality.yml"
    content = workflow_path.read_text()

    assert "render3d_smoke:" in content
    assert "uv sync --extra dev --extra render3d" in content
    assert "tests/test_avatar_renderer_render3d_smoke.py" in content
