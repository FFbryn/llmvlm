import json
from pathlib import Path

from preprocessing.manifest import VisualSample
from preprocessing.visual_manifest import (
    VisualManifestWriter,
)


def test_visual_manifest_writer(tmp_path):
    output_path = (
        tmp_path / "visual_manifest.jsonl"
    )

    samples = [
        VisualSample(
            sample_id="sample_001",
            benchmark="TEST",
            image_path=Path(
                "images/sample_001.png"
            ),
            source_prompt="First test prompt.",
            rendering_config={
                "width": 1024,
                "height": 1024,
            },
        ),
        VisualSample(
            sample_id="sample_002",
            benchmark="TEST",
            image_path=Path(
                "images/sample_002.png"
            ),
            source_prompt="Second test prompt.",
            rendering_config={
                "width": 1024,
                "height": 1024,
            },
        ),
    ]

    writer = VisualManifestWriter(
        output_path=output_path,
    )

    result = writer.write(samples)

    assert result == output_path
    assert output_path.exists()
    assert output_path.is_file()

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["sample_id"] == "sample_001"
    assert second["sample_id"] == "sample_002"

    assert (
        first["benchmark"]
        == "TEST"
    )

    assert (
        first["source_prompt"]
        == "First test prompt."
    )

    assert (
        first["rendering_config"]["width"]
        == 1024
    )