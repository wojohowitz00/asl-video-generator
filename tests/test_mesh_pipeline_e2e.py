"""End-to-end tests for vocabulary -> mesh -> video pipeline."""

import json
from pathlib import Path


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
