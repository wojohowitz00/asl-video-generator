"""Tests for benchmark_mesh_backends script behavior."""

import importlib.util
import json
from pathlib import Path

import pytest


def _load_benchmark_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_mesh_backends.py"
    spec = importlib.util.spec_from_file_location("benchmark_mesh_backends_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_parser_defaults():
    """Benchmark parser should expose stable defaults."""
    module = _load_benchmark_module()
    parser = module.create_parser()
    args = parser.parse_args([])

    assert args.width == 256
    assert args.height == 256
    assert args.fps == 12
    assert args.frames == 8
    assert args.repeats == 3
    assert args.output.name == "mesh_backend_benchmark.json"


def test_run_backend_benchmark_summarizes_timings(monkeypatch, tmp_path):
    """run_backend_benchmark should return timing summary and telemetry fields."""
    module = _load_benchmark_module()

    class _FakeRenderer:
        def __init__(self, config):
            self._backend = config.mesh_backend

        def render_mesh(self, _motion_path, output_path):
            output_path.write_bytes(b"GIF89a")
            return output_path

        def get_mesh_backend_telemetry(self):
            return {
                "last_backend": self._backend,
                "counts": {"stylized": 0, "software_3d": 2, "pyrender": 0},
                "pyrender_fallback_count": 0,
            }

    perf_values = iter([0.0, 0.2, 0.2, 0.5])

    monkeypatch.setattr(module, "AvatarRenderer", _FakeRenderer)
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(perf_values))

    result = module.run_backend_benchmark(
        backend="software_3d",
        repeats=2,
        width=64,
        height=64,
        fps=12,
        sample_motion=module.build_sample_motion(num_frames=2, fps=12),
        workdir=tmp_path,
    )

    assert result["requested_backend"] == "software_3d"
    assert result["effective_last_backend"] == "software_3d"
    assert result["output_exists"] is True
    assert result["timings"]["min_seconds"] == pytest.approx(0.2)
    assert result["timings"]["max_seconds"] == pytest.approx(0.3)
    assert result["timings"]["mean_seconds"] == pytest.approx(0.25)


def test_main_writes_benchmark_json(monkeypatch, tmp_path):
    """main should write a JSON report with both backend sections."""
    module = _load_benchmark_module()

    def _fake_run_backend_benchmark(**kwargs):
        backend = kwargs["backend"]
        return {
            "requested_backend": backend,
            "effective_last_backend": backend,
            "timings": {"min_seconds": 0.1, "mean_seconds": 0.1, "max_seconds": 0.1},
            "pyrender_fallback_count": 0,
            "output_exists": True,
        }

    output_path = tmp_path / "bench.json"
    monkeypatch.setattr(module, "run_backend_benchmark", _fake_run_backend_benchmark)

    module.main(["--output", str(output_path), "--repeats", "1"])

    payload = json.loads(output_path.read_text())
    assert "software_3d" in payload["benchmarks"]
    assert "pyrender" in payload["benchmarks"]
