"""ASL Video Generator - Cloud pipeline for ASL avatar video generation.

A three-stage pipeline for generating photorealistic ASL videos from English text:

1. **Translation**: English → ASL Gloss (using LLM)
2. **Pose Generation**: Gloss → Skeletal Poses (dictionary-based)
3. **Rendering**: Poses → Video (skeleton or diffusion-based)

Optimized for Apple Silicon M4 with MPS backend.

Example usage:
    from asl_video_generator import generate_asl_video

    video_path = generate_asl_video(
        "Hello, how are you?",
        output_path="hello.mp4",
        quality="medium",
    )

CLI usage:
    uv run asl-generate "Hello, how are you?" --output hello.mp4
"""

__version__ = "0.1.0"

from typing import Literal, cast

from .config import (
    DeviceType,
    PipelineConfig,
    QualityPreset,
    QualitySettings,
    create_default_config,
    detect_device,
    load_config_from_env,
    validate_mps_availability,
)
from .diffusion_renderer import DiffusionRenderer, RenderResult
from .gloss_translator import (
    GlossSequence,
    GlossTranslator,
    NonManualMarkers,
    translate_batch,
)
from .pose_dictionary import (
    PoseDictionary,
    PoseKeypoints,
    SignPoseSequence,
    interpolate_poses,
)
from .pose_generator import (
    PoseFrame,
    PoseGenerator,
    PoseSequence,
    generate_poses_batch,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "DeviceType",
    "PipelineConfig",
    "QualityPreset",
    "QualitySettings",
    "create_default_config",
    "detect_device",
    "load_config_from_env",
    "validate_mps_availability",
    # Gloss Translation
    "GlossTranslator",
    "GlossSequence",
    "NonManualMarkers",
    "translate_batch",
    # Pose Dictionary
    "PoseDictionary",
    "PoseKeypoints",
    "SignPoseSequence",
    "interpolate_poses",
    # Pose Generation
    "PoseGenerator",
    "PoseFrame",
    "PoseSequence",
    "generate_poses_batch",
    # Rendering
    "DiffusionRenderer",
    "RenderResult",
    # High-level API
    "generate_asl_video",
]


def generate_asl_video(
    text: str,
    output_path: str = "output.mp4",
    quality: str = "medium",
    provider: str = "openai",
    render_mode: str = "auto",
    reference_image: str | None = None,
) -> str:
    """Generate an ASL video from English text.

    High-level convenience function that runs the full pipeline.

    Args:
        text: English text to translate to ASL video.
        output_path: Path for output video file.
        quality: Quality preset ("preview", "medium", "quality").
        provider: LLM provider ("openai", "gemini", "ollama").
        render_mode: Rendering mode ("skeleton", "diffusion", "auto").
        reference_image: Optional path to reference signer image.

    Returns:
        Path to the generated video file.

    Example:
        >>> from asl_video_generator import generate_asl_video
        >>> video = generate_asl_video("Hello, how are you?", "hello.mp4")
        >>> print(f"Video saved to: {video}")
    """
    from pathlib import Path

    # Configure
    config = load_config_from_env()
    config.quality = QualityPreset(quality)
    if provider not in {"openai", "gemini", "ollama"}:
        raise ValueError(f"Unsupported provider: {provider}")
    provider_value = cast(Literal["openai", "gemini", "ollama"], provider)
    config.llm_provider = provider_value

    if render_mode not in {"diffusion", "skeleton", "auto"}:
        raise ValueError(f"Unsupported render mode: {render_mode}")
    render_mode_value = cast(Literal["diffusion", "skeleton", "auto"], render_mode)

    output = Path(output_path)

    # Stage 1: Translate
    translator = GlossTranslator(provider=provider_value, config=config)
    gloss_seq = translator.translate(text)

    # Stage 2: Generate poses
    pose_gen = PoseGenerator(config=config)
    pose_seq = pose_gen.generate(gloss_seq)

    # Stage 3: Render video
    renderer = DiffusionRenderer(
        config=config,
        reference_image=Path(reference_image) if reference_image else None,
    )
    result = renderer.render(pose_seq, output, mode=render_mode_value)

    return str(result.video_path)
