import json
from pathlib import Path

from preprocessing.visual_manifest_loader import (
    VisualManifestLoader,
)


def test_visual_manifest_loader(tmp_path):
    manifest_path = (
        tmp_path / "visual_manifest.jsonl"
    )

    records = [
        {
            "sample_id": "sample_001",
            "benchmark": "TEST",
            "image_path": "images/sample_001.png",
            "source_prompt": "First prompt.",
            "rendering_config": {
                "width": 1024,
                "height": 1024,
            },
        },
        {
            "sample_id": "sample_002",
            "benchmark": "TEST",
            "image_path": "images/sample_002.png",
            "source_prompt": "Second prompt.",
            "rendering_config": {
                "width": 1024,
                "height": 1024,
            },
        },
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(record)
                + "\n"
            )

    loader = VisualManifestLoader(
        manifest_path
    )

    samples = loader.load()

    assert len(samples) == 2

    assert samples[0].sample_id == "sample_001"
    assert samples[0].benchmark == "TEST"

    assert (
        samples[0].image_path
        == Path("images/sample_001.png")
    )

    assert (
        samples[0].source_prompt
        == "First prompt."
    )

    assert (
        samples[0].rendering_config["width"]
        == 1024
    )


def test_visual_manifest_loader_rejects_missing_file(
    tmp_path,
):
    manifest_path = (
        tmp_path / "missing.jsonl"
    )

    loader = VisualManifestLoader(
        manifest_path
    )

    try:
        loader.load()
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_visual_manifest_loader_rejects_invalid_json(
    tmp_path,
):
    manifest_path = (
        tmp_path / "invalid.jsonl"
    )

    manifest_path.write_text(
        "{invalid json}\n",
        encoding="utf-8",
    )

    loader = VisualManifestLoader(
        manifest_path
    )

    try:
        loader.load()
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "baris 1" in str(exc)