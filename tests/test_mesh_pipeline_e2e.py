"""End-to-end tests for vocabulary -> mesh -> video pipeline."""

import json
from pathlib import Path

import numpy as np


def _write_lite_checkpoint(path: Path) -> Path:
    checkpoint = {
        "model_type": "wsigngen-lite",
        "fps": 24,
        "num_frames": 10,
        "mesh_format": "smplh",
        "body_pose_dim": 72,
        "hand_pose_dim": 45,
        "expression_dim": 10,
        "amplitude": 0.35,
        "translation_scale": 0.03,
        "seed": 17,
    }
    path.write_text(json.dumps(checkpoint))
    return path


def test_vocabulary_to_mesh_video_pipeline(tmp_path):
    """Module-level E2E: model-backed vocabulary motion should render to video."""
    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig
    from asl_video_generator.vocabulary_generator import VocabularyGenerator

    model_path = _write_lite_checkpoint(tmp_path / "wsigngen_lite.json")

    generator = VocabularyGenerator(model_path=model_path, use_placeholder=False)
    motion = generator.generate("hello")

    motion_path = tmp_path / "hello_motion.json"
    motion.save(motion_path)

    renderer = AvatarRenderer(
        RenderConfig(width=128, height=128, fps=24, output_format="gif", avatar_style="mesh")
    )
    output_path = tmp_path / "hello_mesh.gif"

    result = renderer.render_mesh(motion_path, output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_batch_detects_mesh_and_renders_video(tmp_path):
    """render_batch should route mesh json files through render_mesh path."""
    from asl_video_generator.avatar_renderer import RenderConfig, render_batch
    from asl_video_generator.vocabulary_generator import VocabularyGenerator

    model_path = _write_lite_checkpoint(tmp_path / "wsigngen_lite.json")

    input_dir = tmp_path / "motions"
    output_dir = tmp_path / "videos"
    input_dir.mkdir(parents=True, exist_ok=True)

    generator = VocabularyGenerator(model_path=model_path, use_placeholder=False)
    motion = generator.generate("thank you")
    motion.save(input_dir / "thank_you.json")

    results = render_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        config=RenderConfig(
            width=128,
            height=128,
            fps=24,
            output_format="gif",
            avatar_style="mesh",
        ),
    )

    assert len(results) == 1
    assert results[0].suffix == ".gif"
    assert results[0].exists()
    assert results[0].stat().st_size > 0


def test_mesh_video_artifact_characteristics(tmp_path):
    """Rendered mesh GIF should preserve dimensions, frame count, and visible content."""
    import imageio.v2 as imageio

    from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig
    from asl_video_generator.vocabulary_generator import VocabularyGenerator

    model_path = _write_lite_checkpoint(tmp_path / "wsigngen_lite.json")

    generator = VocabularyGenerator(model_path=model_path, use_placeholder=False)
    motion = generator.generate("welcome")

    motion_path = tmp_path / "welcome_motion.json"
    motion.save(motion_path)

    renderer = AvatarRenderer(
        RenderConfig(
            width=160,
            height=120,
            fps=24,
            output_format="gif",
            avatar_style="mesh",
            mesh_backend="software_3d",
        )
    )
    output_path = tmp_path / "welcome_mesh.gif"

    result = renderer.render_mesh(motion_path, output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    frames = imageio.mimread(str(output_path))
    assert len(frames) == len(motion.frames)

    first_frame = np.asarray(frames[0])[..., :3]
    last_frame = np.asarray(frames[-1])[..., :3]

    assert first_frame.shape[:2] == (120, 160)

    non_background_pixels = []
    for frame in frames:
        rgb = np.asarray(frame)[..., :3]
        non_background = np.any(rgb < 245, axis=2)
        non_background_pixels.append(int(non_background.sum()))

    assert min(non_background_pixels) > 0
    assert max(non_background_pixels) > (160 * 120) // 200

    mean_frame_delta = np.mean(
        np.abs(first_frame.astype(np.int16) - last_frame.astype(np.int16))
    )
    assert mean_frame_delta > 0.5
