"""wSignGen-based vocabulary generator for word-level 3D ASL motions.

This module provides word-level 3D mesh motion generation for vocabulary
learning using the wSignGen transformer-diffusion model.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass
class MeshFrame:
    """A single frame of 3D mesh data (SMPL-H format)."""
    
    timestamp_ms: int
    # Body pose (72 params for SMPL, 156 for SMPL-H)
    body_pose: list[float]
    # Hand poses (45 params each for left/right)
    left_hand_pose: list[float]
    right_hand_pose: list[float]
    # Global translation
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Face expression (optional, 10 params)
    expression: list[float] | None = None


@dataclass
class VocabularyMotion:
    """3D motion sequence for a vocabulary word."""
    
    word: str
    asl_gloss: str
    frames: list[MeshFrame]
    fps: int = 30
    total_duration_ms: int = 0
    mesh_format: Literal["smpl", "smplh", "smplx"] = "smplh"
    
    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable format."""
        return {
            "word": self.word,
            "asl_gloss": self.asl_gloss,
            "fps": self.fps,
            "total_duration_ms": self.total_duration_ms,
            "mesh_format": self.mesh_format,
            "frames": [
                {
                    "timestamp_ms": f.timestamp_ms,
                    "body_pose": f.body_pose,
                    "left_hand_pose": f.left_hand_pose,
                    "right_hand_pose": f.right_hand_pose,
                    "translation": f.translation,
                    "expression": f.expression,
                }
                for f in self.frames
            ],
        }
    
    def save(self, path: Path) -> None:
        """Save motion to JSON file."""
        path.write_text(json.dumps(self.to_json(), indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "VocabularyMotion":
        """Load motion from JSON file."""
        data = json.loads(path.read_text())
        frames = [
            MeshFrame(
                timestamp_ms=f["timestamp_ms"],
                body_pose=f["body_pose"],
                left_hand_pose=f["left_hand_pose"],
                right_hand_pose=f["right_hand_pose"],
                translation=tuple(f["translation"]),
                expression=f.get("expression"),
            )
            for f in data["frames"]
        ]
        return cls(
            word=data["word"],
            asl_gloss=data["asl_gloss"],
            frames=frames,
            fps=data["fps"],
            total_duration_ms=data["total_duration_ms"],
            mesh_format=data.get("mesh_format", "smplh"),
        )


class VocabularyGenerator:
    """Generate 3D motions for vocabulary words using wSignGen.
    
    wSignGen uses a transformer-based diffusion model trained on
    word-level ASL videos to generate realistic 3D SMPL-H meshes.
    """
    
    # Common ASL vocabulary with gloss mappings
    COMMON_GLOSSES = {
        "hello": "HELLO",
        "goodbye": "BYE",
        "thank you": "THANK-YOU",
        "please": "PLEASE",
        "sorry": "SORRY",
        "yes": "YES",
        "no": "NO",
        "help": "HELP",
        "water": "WATER",
        "food": "FOOD",
        "bathroom": "BATHROOM",
        "name": "NAME",
        "understand": "UNDERSTAND",
        "like": "LIKE",
        "want": "WANT",
        "need": "NEED",
        "good": "GOOD",
        "bad": "BAD",
        "more": "MORE",
        "done": "DONE",
    }

    _MESH_FORMATS: set[str] = {"smpl", "smplh", "smplx"}
    
    def __init__(
        self,
        model_path: str | Path | None = None,
        use_placeholder: bool = True,
    ):
        """Initialize vocabulary generator.
        
        Args:
            model_path: Path to wSignGen model weights
            use_placeholder: If True, use placeholder motions
        """
        self.model_path = Path(model_path) if model_path is not None else None
        self.use_placeholder = use_placeholder
        self._model = None
    
    def generate(self, word: str) -> VocabularyMotion:
        """Generate 3D motion for a vocabulary word.
        
        Args:
            word: English word to generate ASL sign for
            
        Returns:
            VocabularyMotion with 3D mesh animation data
        """
        gloss = self.COMMON_GLOSSES.get(word.lower(), word.upper())
        
        if self.use_placeholder:
            return self._generate_placeholder(word, gloss)
        else:
            return self._generate_wsigngen(word, gloss)
    
    def _generate_placeholder(self, word: str, gloss: str) -> VocabularyMotion:
        """Generate placeholder motion for testing.
        
        Creates a simple hand movement animation.
        """
        fps = 30
        duration_ms = 1500  # 1.5 seconds per word
        num_frames = int(duration_ms / 1000 * fps)
        
        frames = []
        for i in range(num_frames):
            timestamp = int(i / fps * 1000)
            t = i / num_frames
            
            # Neutral body pose (72 params for SMPL)
            body_pose = [0.0] * 72
            # Add slight shoulder movement
            body_pose[48] = np.sin(t * np.pi * 2) * 0.3  # Right shoulder
            body_pose[51] = np.sin(t * np.pi * 2 + 0.5) * 0.2  # Left shoulder
            
            # Hand poses (45 params each - 15 joints * 3 rotations)
            # Simulate finger movements
            left_hand = [0.0] * 45
            right_hand = [np.sin(t * np.pi * 4 + j * 0.1) * 0.5 for j in range(45)]
            
            frames.append(MeshFrame(
                timestamp_ms=timestamp,
                body_pose=body_pose,
                left_hand_pose=left_hand,
                right_hand_pose=right_hand,
                translation=(0.0, 0.0, 0.0),
            ))
        
        return VocabularyMotion(
            word=word,
            asl_gloss=gloss,
            frames=frames,
            fps=fps,
            total_duration_ms=duration_ms,
        )
    
    def _generate_wsigngen(self, word: str, gloss: str) -> VocabularyMotion:
        """Generate motion using wSignGen model.
        """
        if self._model is None:
            self._load_model()

        if isinstance(self._model, dict) and self._model.get("model_type") == "wsigngen-lite":
            return self._infer_from_lite_checkpoint(word, gloss, self._model)

        result = self._run_external_model(word, gloss)
        if isinstance(result, VocabularyMotion):
            return result
        if not isinstance(result, dict):
            raise TypeError("wSignGen model output must be VocabularyMotion or dict")
        return self._motion_from_output_dict(word, gloss, result)

    def _load_model(self) -> None:
        """Load wSignGen model weights."""
        if self.model_path is None:
            raise ValueError("model_path is required when use_placeholder=False")

        if not self.model_path.exists():
            raise FileNotFoundError(f"wSignGen model not found: {self.model_path}")

        if self.model_path.suffix.lower() == ".json":
            model_data = json.loads(self.model_path.read_text())
            model_type = model_data.get("model_type", "wsigngen-lite")
            if model_type != "wsigngen-lite":
                raise ValueError(f"Unsupported JSON model_type: {model_type}")
            self._validate_lite_checkpoint(model_data)
            self._model = model_data
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Loading non-JSON wSignGen checkpoints requires torch to be installed"
            ) from exc

        self._model = torch.load(self.model_path, map_location="cpu")

    def _validate_lite_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Validate required fields for lightweight checkpoint format."""
        fps = int(checkpoint.get("fps", 30))
        num_frames = int(checkpoint.get("num_frames", 45))
        body_pose_dim = int(checkpoint.get("body_pose_dim", 72))
        hand_pose_dim = int(checkpoint.get("hand_pose_dim", 45))
        expression_dim = int(checkpoint.get("expression_dim", 10))
        mesh_format = str(checkpoint.get("mesh_format", "smplh"))

        if fps <= 0:
            raise ValueError("Lite checkpoint field fps must be > 0")
        if num_frames <= 0:
            raise ValueError("Lite checkpoint field num_frames must be > 0")
        if body_pose_dim <= 0:
            raise ValueError("Lite checkpoint field body_pose_dim must be > 0")
        if hand_pose_dim <= 0:
            raise ValueError("Lite checkpoint field hand_pose_dim must be > 0")
        if expression_dim < 0:
            raise ValueError("Lite checkpoint field expression_dim must be >= 0")
        if mesh_format not in self._MESH_FORMATS:
            raise ValueError("Lite checkpoint field mesh_format must be one of smpl/smplh/smplx")

    def _infer_from_lite_checkpoint(
        self,
        word: str,
        gloss: str,
        checkpoint: dict[str, Any],
    ) -> VocabularyMotion:
        """Run deterministic lightweight inference from JSON checkpoint parameters."""
        fps = int(checkpoint.get("fps", 30))
        num_frames = int(checkpoint.get("num_frames", 45))
        body_pose_dim = int(checkpoint.get("body_pose_dim", 72))
        hand_pose_dim = int(checkpoint.get("hand_pose_dim", 45))
        expression_dim = int(checkpoint.get("expression_dim", 10))
        mesh_format = self._normalize_mesh_format(str(checkpoint.get("mesh_format", "smplh")))
        amplitude = float(checkpoint.get("amplitude", 0.3))
        translation_scale = float(checkpoint.get("translation_scale", 0.02))
        base_seed = int(checkpoint.get("seed", 0))

        token = f"{word}|{gloss}".encode()
        digest = hashlib.sha256(token).digest()
        token_seed = int.from_bytes(digest[:4], byteorder="little")
        rng = np.random.default_rng(base_seed + token_seed)

        body_phase = rng.uniform(0.0, 2.0 * np.pi, size=body_pose_dim)
        left_phase = rng.uniform(0.0, 2.0 * np.pi, size=hand_pose_dim)
        right_phase = rng.uniform(0.0, 2.0 * np.pi, size=hand_pose_dim)
        expression_phase = rng.uniform(0.0, 2.0 * np.pi, size=expression_dim)

        frame_duration_ms = 1000.0 / fps
        frames: list[MeshFrame] = []
        for i in range(num_frames):
            t = i / num_frames
            timestamp_ms = int(round(i * frame_duration_ms))

            body_pose = (
                amplitude * np.sin(2.0 * np.pi * t + body_phase)
            ).astype(np.float64).tolist()
            left_hand = (
                amplitude * np.sin(3.0 * np.pi * t + left_phase)
            ).astype(np.float64).tolist()
            right_hand = (
                amplitude * np.cos(3.0 * np.pi * t + right_phase)
            ).astype(np.float64).tolist()

            expression: list[float] | None = None
            if expression_dim > 0:
                expression = (
                    0.5 * amplitude * np.sin(2.0 * np.pi * t + expression_phase)
                ).astype(np.float64).tolist()

            translation = (
                float(translation_scale * np.sin(2.0 * np.pi * t)),
                0.0,
                float(translation_scale * np.cos(2.0 * np.pi * t)),
            )

            frames.append(
                MeshFrame(
                    timestamp_ms=timestamp_ms,
                    body_pose=body_pose,
                    left_hand_pose=left_hand,
                    right_hand_pose=right_hand,
                    translation=translation,
                    expression=expression,
                )
            )

        total_duration_ms = int(round(num_frames * frame_duration_ms))
        return VocabularyMotion(
            word=word,
            asl_gloss=gloss,
            frames=frames,
            fps=fps,
            total_duration_ms=total_duration_ms,
            mesh_format=mesh_format,
        )

    def _run_external_model(self, word: str, gloss: str) -> Any:
        """Execute model inference for non-lite checkpoints."""
        model = self._model
        if model is None:
            raise RuntimeError("wSignGen model is not loaded")

        generate_fn = getattr(model, "generate", None)
        if generate_fn is None:
            if not callable(model):
                raise TypeError("Loaded wSignGen model is not callable and has no generate()")
            generate_fn = model

        for kwargs in (
            {"word": word, "gloss": gloss},
            {"gloss": gloss},
            {"word": word},
            {},
        ):
            try:
                return generate_fn(**kwargs)
            except TypeError:
                continue

        # Fallback positional calls.
        try:
            return generate_fn(word, gloss)
        except TypeError:
            return generate_fn(gloss)

    def _motion_from_output_dict(
        self,
        word: str,
        gloss: str,
        output: dict[str, Any],
    ) -> VocabularyMotion:
        """Convert dictionary-based model output into VocabularyMotion."""
        if "frames" not in output:
            raise ValueError("Model output dict must contain a 'frames' list")

        fps = int(output.get("fps", 30))
        if fps <= 0:
            raise ValueError("Model output fps must be > 0")

        mesh_format = self._normalize_mesh_format(str(output.get("mesh_format", "smplh")))

        frames_data = output["frames"]
        if not isinstance(frames_data, list):
            raise ValueError("Model output 'frames' must be a list")

        parsed_frames: list[MeshFrame] = []
        for index, frame in enumerate(frames_data):
            if not isinstance(frame, dict):
                raise ValueError("Each model output frame must be a dict")

            timestamp_ms = int(frame.get("timestamp_ms", round(index * (1000.0 / fps))))
            body_pose = [float(v) for v in frame["body_pose"]]
            left_hand_pose = [float(v) for v in frame["left_hand_pose"]]
            right_hand_pose = [float(v) for v in frame["right_hand_pose"]]
            translation_data = frame.get("translation", (0.0, 0.0, 0.0))
            if not isinstance(translation_data, (list, tuple)) or len(translation_data) != 3:
                raise ValueError("Frame translation must be a 3-value tuple/list")
            translation = (
                float(translation_data[0]),
                float(translation_data[1]),
                float(translation_data[2]),
            )
            expression_raw = frame.get("expression")
            expression = (
                [float(v) for v in expression_raw]
                if isinstance(expression_raw, list)
                else None
            )

            parsed_frames.append(
                MeshFrame(
                    timestamp_ms=timestamp_ms,
                    body_pose=body_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    translation=translation,
                    expression=expression,
                )
            )

        if not parsed_frames:
            raise ValueError("Model output 'frames' must not be empty")

        total_duration_ms = int(
            output.get("total_duration_ms", round(len(parsed_frames) * (1000.0 / fps)))
        )
        return VocabularyMotion(
            word=word,
            asl_gloss=gloss,
            frames=parsed_frames,
            fps=fps,
            total_duration_ms=total_duration_ms,
            mesh_format=mesh_format,
        )

    def _normalize_mesh_format(
        self, mesh_format: str
    ) -> Literal["smpl", "smplh", "smplx"]:
        """Validate and narrow mesh format literal type."""
        if mesh_format not in self._MESH_FORMATS:
            raise ValueError("mesh_format must be one of smpl/smplh/smplx")
        if mesh_format == "smpl":
            return "smpl"
        if mesh_format == "smplh":
            return "smplh"
        return "smplx"


def generate_vocabulary_batch(
    words: list[str],
    output_dir: Path,
    use_placeholder: bool = True,
) -> list[Path]:
    """Generate 3D motions for a batch of vocabulary words.
    
    Args:
        words: List of English words
        output_dir: Directory to save motion JSON files
        use_placeholder: If True, use placeholder motions
        
    Returns:
        List of paths to generated motion files
    """
    generator = VocabularyGenerator(use_placeholder=use_placeholder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for word in words:
        motion = generator.generate(word)
        output_path = output_dir / f"{word.lower().replace(' ', '_')}.json"
        motion.save(output_path)
        paths.append(output_path)
        print(f"Generated vocabulary: {word} -> {output_path}")
    
    return paths
