"""Tests for mesh rendering pipeline in AvatarRenderer."""

import json


def _sample_mesh_motion(num_frames: int = 6) -> dict:
    frames = []
    for i in range(num_frames):
        frames.append(
            {
                "timestamp_ms": i * 40,
                "body_pose": [0.01 * (j + i) for j in range(72)],
                "left_hand_pose": [0.02 * (j + i) for j in range(45)],
                "right_hand_pose": [0.03 * (j + i) for j in range(45)],
                "translation": [0.0, 0.0, 0.0],
                "expression": [0.001 * (j + i) for j in range(10)],
            }
        )

    return {
        "word": "hello",
        "asl_gloss": "HELLO",
        "fps": 24,
        "total_duration_ms": num_frames * 40,
        "mesh_format": "smplh",
        "frames": frames,
    }


def test_render_mesh_creates_video_file(tmp_path):
    """render_mesh should write a video file, not only frame directory."""
    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    motion_data = _sample_mesh_motion(num_frames=5)
    motion_path = tmp_path / "motion.json"
    motion_path.write_text(json.dumps(motion_data))

    output_path = tmp_path / "mesh.gif"
    renderer = AvatarRenderer(
        RenderConfig(width=128, height=128, fps=24, output_format="gif", avatar_style="mesh")
    )

    result = renderer.render_mesh(motion_path, output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_mesh_falls_back_to_frames_if_writer_fails(tmp_path, monkeypatch):
    """When video writer fails, renderer should still export frame sequence."""
    import imageio

    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    motion_data = _sample_mesh_motion(num_frames=4)
    motion_path = tmp_path / "motion.json"
    motion_path.write_text(json.dumps(motion_data))

    def _raise_writer_error(*_args, **_kwargs):
        raise RuntimeError("writer failed")

    monkeypatch.setattr(imageio, "get_writer", _raise_writer_error)

    output_path = tmp_path / "mesh.gif"
    renderer = AvatarRenderer(
        RenderConfig(width=128, height=128, fps=24, output_format="gif", avatar_style="mesh")
    )

    result = renderer.render_mesh(motion_path, output_path)
    frames_dir = output_path.parent / f"{output_path.stem}_frames"

    assert result == frames_dir
    assert frames_dir.exists()
    assert len(list(frames_dir.glob("frame_*.png"))) == len(motion_data["frames"])
