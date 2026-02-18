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


def _sample_vertices_motion(num_frames: int = 6) -> dict:
    base_vertices = [
        (-0.45, -0.45, 0.0),
        (0.45, -0.45, 0.0),
        (0.0, 0.45, 0.0),
        (0.0, -0.1, 0.55),
    ]
    faces = [
        (0, 1, 2),
        (0, 1, 3),
        (1, 2, 3),
        (2, 0, 3),
    ]

    frames = []
    for i in range(num_frames):
        phase = i / max(num_frames - 1, 1)
        z_offset = 0.12 * phase
        vertices = [[x, y, z + z_offset] for x, y, z in base_vertices]
        frames.append(
            {
                "timestamp_ms": i * 40,
                "vertices": vertices,
                "faces": [list(face) for face in faces],
                "translation": [0.0, 0.0, 0.0],
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


def test_render_config_accepts_3d_mesh_backend():
    """RenderConfig should expose an explicit 3D mesh backend option."""
    from asl_video_generator.avatar_renderer import RenderConfig

    config = RenderConfig(avatar_style="mesh", output_format="gif", mesh_backend="software_3d")

    assert config.mesh_backend == "software_3d"


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


def test_render_mesh_3d_backend_responds_to_camera_angle(tmp_path):
    """3D backend output should change when camera angle changes."""
    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    motion_data = _sample_vertices_motion(num_frames=6)
    motion_path = tmp_path / "motion_vertices.json"
    motion_path.write_text(json.dumps(motion_data))

    out_a = tmp_path / "mesh_cam_a.gif"
    out_b = tmp_path / "mesh_cam_b.gif"

    renderer_a = AvatarRenderer(
        RenderConfig(
            width=128,
            height=128,
            fps=24,
            output_format="gif",
            avatar_style="mesh",
            mesh_backend="software_3d",
            camera_angle=(0.0, 0.0),
        )
    )
    renderer_b = AvatarRenderer(
        RenderConfig(
            width=128,
            height=128,
            fps=24,
            output_format="gif",
            avatar_style="mesh",
            mesh_backend="software_3d",
            camera_angle=(45.0, 20.0),
        )
    )

    result_a = renderer_a.render_mesh(motion_path, out_a)
    result_b = renderer_b.render_mesh(motion_path, out_b)

    assert result_a == out_a
    assert result_b == out_b
    assert out_a.exists()
    assert out_b.exists()
    assert out_a.read_bytes() != out_b.read_bytes()


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


def test_pyrender_backend_falls_back_to_software_when_unavailable(monkeypatch, capsys):
    """pyrender backend should fall back to software_3d when dependencies are unavailable."""
    from PIL import Image

    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    renderer = AvatarRenderer(
        RenderConfig(avatar_style="mesh", output_format="gif", mesh_backend="pyrender")
    )

    software_calls = {"count": 0}

    def _fake_software(*_args, **_kwargs):
        software_calls["count"] += 1
        return Image.new("RGB", (32, 32), (255, 255, 255))

    def _fake_pyrender(*_args, **_kwargs):
        raise AssertionError("pyrender path should not execute when unavailable")

    monkeypatch.setattr(renderer, "_is_pyrender_available", lambda: False)
    monkeypatch.setattr(renderer, "_build_mesh_image_software_3d", _fake_software)
    monkeypatch.setattr(renderer, "_build_mesh_image_pyrender", _fake_pyrender)

    renderer._build_mesh_image({}, 0, 1)
    renderer._build_mesh_image({}, 1, 1)

    captured = capsys.readouterr()
    assert software_calls["count"] == 2
    assert captured.out.count("pyrender backend unavailable") == 1
    telemetry = renderer.get_mesh_backend_telemetry()
    assert telemetry["last_backend"] == "software_3d"
    assert telemetry["counts"]["software_3d"] == 2
    assert telemetry["counts"]["pyrender"] == 0
    assert telemetry["pyrender_fallback_count"] == 2


def test_pyrender_backend_routes_to_pyrender_path_when_available(monkeypatch):
    """pyrender backend should use pyrender route when dependency check passes."""
    from PIL import Image

    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    renderer = AvatarRenderer(
        RenderConfig(avatar_style="mesh", output_format="gif", mesh_backend="pyrender")
    )

    pyrender_calls = {"count": 0}

    def _fake_pyrender(*_args, **_kwargs):
        pyrender_calls["count"] += 1
        return Image.new("RGB", (32, 32), (240, 240, 240))

    def _fake_software(*_args, **_kwargs):
        raise AssertionError("software fallback should not execute when pyrender is available")

    monkeypatch.setattr(renderer, "_is_pyrender_available", lambda: True)
    monkeypatch.setattr(renderer, "_build_mesh_image_pyrender", _fake_pyrender)
    monkeypatch.setattr(renderer, "_build_mesh_image_software_3d", _fake_software)

    renderer._build_mesh_image({}, 0, 1)

    assert pyrender_calls["count"] == 1
    telemetry = renderer.get_mesh_backend_telemetry()
    assert telemetry["last_backend"] == "pyrender"
    assert telemetry["counts"]["pyrender"] == 1
    assert telemetry["counts"]["software_3d"] == 0
    assert telemetry["pyrender_fallback_count"] == 0


def test_pyrender_backend_falls_back_when_native_renderer_errors(monkeypatch, capsys):
    """pyrender backend should fall back to software_3d when native render fails."""
    from PIL import Image

    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    renderer = AvatarRenderer(
        RenderConfig(avatar_style="mesh", output_format="gif", mesh_backend="pyrender")
    )

    software_calls = {"count": 0}

    def _failing_pyrender(*_args, **_kwargs):
        raise RuntimeError("offscreen init failed")

    def _fake_software(*_args, **_kwargs):
        software_calls["count"] += 1
        return Image.new("RGB", (24, 24), (200, 200, 200))

    monkeypatch.setattr(renderer, "_is_pyrender_available", lambda: True)
    monkeypatch.setattr(renderer, "_build_mesh_image_pyrender", _failing_pyrender)
    monkeypatch.setattr(renderer, "_build_mesh_image_software_3d", _fake_software)

    renderer._build_mesh_image({}, 0, 1)
    renderer._build_mesh_image({}, 1, 1)

    captured = capsys.readouterr()
    assert software_calls["count"] == 2
    assert captured.out.count("pyrender backend failed") == 1
    telemetry = renderer.get_mesh_backend_telemetry()
    assert telemetry["last_backend"] == "software_3d"
    assert telemetry["counts"]["software_3d"] == 2
    assert telemetry["counts"]["pyrender"] == 0
    assert telemetry["pyrender_fallback_count"] == 2


def test_build_mesh_image_pyrender_uses_native_rgb_array(monkeypatch):
    """_build_mesh_image_pyrender should convert native RGB output to a PIL image."""
    import numpy as np

    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig

    renderer = AvatarRenderer(
        RenderConfig(avatar_style="mesh", output_format="gif", mesh_backend="pyrender")
    )

    vertices = np.array(
        [
            [-0.2, -0.2, 0.0],
            [0.2, -0.2, 0.0],
            [0.0, 0.25, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    rgb = np.full((18, 22, 3), 123, dtype=np.uint8)

    monkeypatch.setattr(renderer, "_extract_or_synthesize_mesh", lambda _frame: (vertices, faces))
    monkeypatch.setattr(renderer, "_render_with_pyrender", lambda _v, _f, _frame: rgb)

    image = renderer._build_mesh_image_pyrender({}, 0, 1)

    assert image.size == (22, 18)
