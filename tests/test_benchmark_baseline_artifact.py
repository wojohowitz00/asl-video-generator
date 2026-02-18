"""Schema validation for committed benchmark baseline artifact."""

import json
from pathlib import Path


def test_benchmark_baseline_artifact_has_expected_schema():
    """Committed benchmark baseline JSON should expose stable report keys."""
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmarks"
        / "mesh_backend_benchmark_baseline.json"
    )
    assert artifact_path.exists()

    payload = json.loads(artifact_path.read_text())
    assert "generated_at" in payload
    assert isinstance(payload["generated_at"], str)

    config = payload["config"]
    assert set(config.keys()) == {"width", "height", "fps", "frames", "repeats"}
    for key in ("width", "height", "fps", "frames", "repeats"):
        assert isinstance(config[key], int)
        assert config[key] > 0

    benchmarks = payload["benchmarks"]
    assert set(benchmarks.keys()) == {"software_3d", "pyrender"}

    for backend_name, section in benchmarks.items():
        assert section["requested_backend"] == backend_name
        assert section["effective_last_backend"] in {"software_3d", "pyrender", "stylized", None}
        assert isinstance(section["output_exists"], bool)
        assert isinstance(section["pyrender_fallback_count"], int)
        assert section["pyrender_fallback_count"] >= 0

        counts = section["effective_counts"]
        assert set(counts.keys()) == {"stylized", "software_3d", "pyrender"}
        for value in counts.values():
            assert isinstance(value, int)
            assert value >= 0

        timings = section["timings"]
        assert set(timings.keys()) == {"min_seconds", "mean_seconds", "max_seconds"}
        assert timings["min_seconds"] >= 0.0
        assert timings["mean_seconds"] >= 0.0
        assert timings["max_seconds"] >= 0.0
        assert timings["min_seconds"] <= timings["mean_seconds"] <= timings["max_seconds"]
