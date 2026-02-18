#!/usr/bin/env python3
"""Benchmark mesh renderer backends on a fixed synthetic motion sample."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from asl_video_generator.avatar_renderer import AvatarRenderer, RenderConfig


def create_parser() -> argparse.ArgumentParser:
    """Create CLI parser for mesh backend benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark mesh rendering backends")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/benchmarks/mesh_backend_benchmark.json"),
        help="Path to write benchmark JSON output",
    )
    parser.add_argument("--width", type=int, default=256, help="Render width")
    parser.add_argument("--height", type=int, default=256, help="Render height")
    parser.add_argument("--fps", type=int, default=12, help="Render fps")
    parser.add_argument("--frames", type=int, default=8, help="Number of synthetic sample frames")
    parser.add_argument("--repeats", type=int, default=3, help="Benchmark repeats per backend")
    return parser


def build_sample_motion(num_frames: int, fps: int) -> dict[str, Any]:
    """Build fixed synthetic motion payload for repeatable benchmarking."""
    frames: list[dict[str, Any]] = []
    for i in range(max(num_frames, 1)):
        phase = i / max(num_frames - 1, 1)
        frames.append(
            {
                "timestamp_ms": int((1000 / max(fps, 1)) * i),
                "vertices": [
                    [-0.40 + 0.02 * phase, -0.40, 0.00],
                    [0.40 + 0.02 * phase, -0.40, 0.00],
                    [0.02 * phase, 0.45, 0.05 * phase],
                ],
                "faces": [[0, 1, 2]],
                "translation": [0.0, 0.0, 0.02 * phase],
            }
        )

    return {
        "word": "benchmark",
        "asl_gloss": "BENCHMARK",
        "fps": max(fps, 1),
        "frames": frames,
    }


def _summarize_timings(samples: list[float]) -> dict[str, float]:
    """Compute min/mean/max timing summary."""
    if not samples:
        return {"min_seconds": 0.0, "mean_seconds": 0.0, "max_seconds": 0.0}
    return {
        "min_seconds": min(samples),
        "mean_seconds": sum(samples) / len(samples),
        "max_seconds": max(samples),
    }


def run_backend_benchmark(
    *,
    backend: str,
    repeats: int,
    width: int,
    height: int,
    fps: int,
    sample_motion: dict[str, Any] | None = None,
    workdir: Path | None = None,
) -> dict[str, Any]:
    """Benchmark a specific backend and return summary payload."""
    motion_data = sample_motion or build_sample_motion(num_frames=8, fps=fps)
    timings: list[float] = []
    final_result_path: Path | None = None
    output_created = False
    final_telemetry: dict[str, Any] = {
        "last_backend": None,
        "counts": {"stylized": 0, "software_3d": 0, "pyrender": 0},
        "pyrender_fallback_count": 0,
    }

    if workdir is not None:
        workdir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=workdir) as tmpdir:
        tmp_path = Path(tmpdir)
        motion_path = tmp_path / "motion.json"
        motion_path.write_text(json.dumps(motion_data))

        for idx in range(max(repeats, 1)):
            output_path = tmp_path / f"{backend}_{idx}.gif"
            renderer = AvatarRenderer(
                RenderConfig(
                    width=width,
                    height=height,
                    fps=fps,
                    output_format="gif",
                    avatar_style="mesh",
                    mesh_backend=backend,
                )
            )

            start = time.perf_counter()
            result = renderer.render_mesh(motion_path, output_path)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            final_result_path = result
            output_created = result.exists()
            final_telemetry = renderer.get_mesh_backend_telemetry()

    result_exists = bool(final_result_path is not None and output_created)
    return {
        "requested_backend": backend,
        "effective_last_backend": final_telemetry["last_backend"],
        "effective_counts": final_telemetry["counts"],
        "pyrender_fallback_count": final_telemetry["pyrender_fallback_count"],
        "timings": _summarize_timings(timings),
        "output_exists": result_exists,
    }


def main(argv: list[str] | None = None) -> None:
    """Run benchmark and write JSON report."""
    parser = create_parser()
    args = parser.parse_args(argv)

    sample_motion = build_sample_motion(num_frames=args.frames, fps=args.fps)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "frames": args.frames,
            "repeats": args.repeats,
        },
        "benchmarks": {
            "software_3d": run_backend_benchmark(
                backend="software_3d",
                repeats=args.repeats,
                width=args.width,
                height=args.height,
                fps=args.fps,
                sample_motion=sample_motion,
                workdir=args.output.parent,
            ),
            "pyrender": run_backend_benchmark(
                backend="pyrender",
                repeats=args.repeats,
                width=args.width,
                height=args.height,
                fps=args.fps,
                sample_motion=sample_motion,
                workdir=args.output.parent,
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Benchmark report saved to: {args.output}")


if __name__ == "__main__":
    main()
