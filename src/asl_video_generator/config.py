"""M4/MPS-optimized configuration for ASL video generation.

This module handles device detection, memory optimization, and quality presets
tailored for Apple Silicon with unified memory architecture.
"""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast


class DeviceType(Enum):
    """Available compute backends."""

    MPS = "mps"      # Apple Silicon GPU
    CUDA = "cuda"    # NVIDIA GPU
    CPU = "cpu"      # CPU fallback


class QualityPreset(Enum):
    """Quality presets balancing speed vs. output quality."""

    PREVIEW = "preview"    # Fast iteration, lower quality
    MEDIUM = "medium"      # Balanced (default)
    QUALITY = "quality"    # High quality, slower


@dataclass(frozen=True)
class QualitySettings:
    """Resolution and generation settings for a quality preset."""

    width: int
    height: int
    fps: int
    diffusion_steps: int
    guidance_scale: float
    # Chunk duration in seconds for progressive rendering
    chunk_duration: float = 2.5


# Quality preset definitions
QUALITY_PRESETS: dict[QualityPreset, QualitySettings] = {
    QualityPreset.PREVIEW: QualitySettings(
        width=384,
        height=672,
        fps=15,
        diffusion_steps=12,
        guidance_scale=5.0,
        chunk_duration=2.0,
    ),
    QualityPreset.MEDIUM: QualitySettings(
        width=576,
        height=1024,
        fps=20,
        diffusion_steps=20,
        guidance_scale=7.5,
        chunk_duration=2.5,
    ),
    QualityPreset.QUALITY: QualitySettings(
        width=720,
        height=1280,
        fps=24,
        diffusion_steps=25,
        guidance_scale=7.5,
        chunk_duration=3.0,
    ),
}


@dataclass
class PipelineConfig:
    """Configuration for the ASL video generation pipeline."""

    # Device settings
    device: DeviceType = field(default_factory=lambda: detect_device())  # type: ignore[has-type]
    use_fp16: bool = True  # Half precision for memory efficiency

    # Quality preset
    quality: QualityPreset = QualityPreset.MEDIUM

    # Memory optimization (relevant for 24GB unified memory)
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_model_cpu_offload: bool = False  # Only for low VRAM
    max_batch_size: int = 1  # Keep at 1 for video generation

    # Paths
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "asl-video")
    pose_dictionary_path: Path | None = None
    models_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "asl-video" / "models"
    )

    # Translation settings
    llm_provider: Literal["openai", "gemini", "ollama"] = "openai"
    llm_model: str = "gpt-4o"  # or "gemini-1.5-flash" or "llama3.2"
    enable_translation_cache: bool = True

    # Pose generation settings
    pose_fps: int = 30
    blend_frames: int = 20  # Frames for interpolation between signs

    # Diffusion settings (can override quality preset)
    custom_width: int | None = None
    custom_height: int | None = None
    custom_fps: int | None = None

    @property
    def torch_dtype(self) -> Any:
        """Get appropriate torch dtype based on settings."""
        import torch
        return torch.float16 if self.use_fp16 else torch.float32

    @property
    def torch_device(self) -> str:
        """Get torch device string."""
        return self.device.value

    @property
    def settings(self) -> QualitySettings:
        """Get quality settings, with custom overrides applied."""
        base = QUALITY_PRESETS[self.quality]
        return QualitySettings(
            width=self.custom_width or base.width,
            height=self.custom_height or base.height,
            fps=self.custom_fps or base.fps,
            diffusion_steps=base.diffusion_steps,
            guidance_scale=base.guidance_scale,
            chunk_duration=base.chunk_duration,
        )

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def detect_device() -> DeviceType:
    """Detect the best available compute device.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU

    Returns:
        DeviceType: The detected device type.
    """
    try:
        import torch

        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return DeviceType.MPS

        # Check for CUDA (NVIDIA)
        if torch.cuda.is_available():
            return DeviceType.CUDA

    except ImportError:
        pass

    return DeviceType.CPU


def validate_mps_availability() -> dict[str, Any]:
    """Validate MPS availability and return diagnostic info.

    Returns:
        Dict with device info and any warnings.
    """
    result: dict[str, Any] = {
        "device": "unknown",
        "available": False,
        "warnings": [],
        "info": {},
    }

    try:
        import torch

        result["info"]["torch_version"] = torch.__version__
        result["info"]["python_version"] = (
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )

        # Check MPS
        if hasattr(torch.backends, "mps"):
            result["info"]["mps_built"] = torch.backends.mps.is_built()
            result["info"]["mps_available"] = torch.backends.mps.is_available()

            if torch.backends.mps.is_available():
                result["device"] = "mps"
                result["available"] = True

                # Test with a small tensor operation
                try:
                    x = torch.tensor([1.0, 2.0, 3.0], device="mps")
                    y = x * 2
                    _ = y.cpu()  # Ensure operation completes
                    result["info"]["mps_functional"] = True
                except Exception as e:
                    result["warnings"].append(f"MPS tensor test failed: {e}")
                    result["info"]["mps_functional"] = False
            else:
                result["warnings"].append("MPS not available on this system")

        # Check CUDA as fallback
        if not result["available"] and torch.cuda.is_available():
            result["device"] = "cuda"
            result["available"] = True
            result["info"]["cuda_device"] = torch.cuda.get_device_name(0)

        # CPU fallback
        if not result["available"]:
            result["device"] = "cpu"
            result["available"] = True
            result["warnings"].append("Using CPU - generation will be slow")

    except ImportError:
        result["warnings"].append("PyTorch not installed")

    return result


def get_memory_info() -> dict[str, Any]:
    """Get system memory information.

    Returns:
        Dict with memory stats relevant for model loading.
    """
    import platform

    info: dict[str, Any] = {
        "platform": platform.system(),
        "machine": platform.machine(),
    }

    try:
        import psutil
        vm = psutil.virtual_memory()
        info["total_gb"] = round(vm.total / (1024**3), 1)
        info["available_gb"] = round(vm.available / (1024**3), 1)
        info["used_percent"] = vm.percent
    except ImportError:
        info["note"] = "Install psutil for memory info"

    # Apple Silicon specific
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        info["apple_silicon"] = True
        info["unified_memory"] = True
        # On M4, unified memory is shared between CPU and GPU
        info["note"] = "Unified memory - GPU and CPU share the same RAM pool"

    return info


def create_default_config() -> PipelineConfig:
    """Create a default configuration optimized for the current system.

    Returns:
        PipelineConfig: Optimized configuration.
    """
    config = PipelineConfig()

    # Adjust based on device
    if config.device == DeviceType.CPU:
        # CPU mode - reduce quality to make it usable
        config.quality = QualityPreset.PREVIEW
        config.use_fp16 = False  # CPU doesn't benefit from FP16
        config.enable_model_cpu_offload = False
    elif config.device == DeviceType.MPS:
        # MPS optimizations
        config.use_fp16 = True
        config.enable_attention_slicing = True
        config.enable_vae_slicing = True

    config.ensure_directories()
    return config


# Environment variable overrides
def load_config_from_env() -> PipelineConfig:
    """Load configuration with environment variable overrides.

    Environment variables:
        ASL_DEVICE: Force device (mps, cuda, cpu)
        ASL_QUALITY: Quality preset (preview, medium, quality)
        ASL_LLM_PROVIDER: LLM provider (openai, gemini, ollama)
        ASL_CACHE_DIR: Cache directory path
        ASL_FP16: Use FP16 (true/false)

    Returns:
        PipelineConfig: Configuration with env overrides.
    """
    config = create_default_config()

    # Device override
    if device_env := os.getenv("ASL_DEVICE"):
        try:
            config.device = DeviceType(device_env.lower())
        except ValueError:
            pass

    # Quality override
    if quality_env := os.getenv("ASL_QUALITY"):
        try:
            config.quality = QualityPreset(quality_env.lower())
        except ValueError:
            pass

    # LLM provider override
    if llm_env := os.getenv("ASL_LLM_PROVIDER"):
        provider = llm_env.lower()
        if provider in ("openai", "gemini", "ollama"):
            config.llm_provider = cast(Literal["openai", "gemini", "ollama"], provider)

    # Cache dir override
    if cache_env := os.getenv("ASL_CACHE_DIR"):
        config.cache_dir = Path(cache_env)
        config.models_dir = config.cache_dir / "models"

    # FP16 override
    if fp16_env := os.getenv("ASL_FP16"):
        config.use_fp16 = fp16_env.lower() in ("true", "1", "yes")

    config.ensure_directories()
    return config
