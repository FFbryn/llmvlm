from pathlib import Path

import pytest

from evaluation.input_builder import ExperimentInputBuilder


def test_build_llm_text():
    builder = ExperimentInputBuilder()

    prompt = "Test benchmark prompt"

    result = builder.build_llm_text(prompt)

    assert result["prompt"] == prompt
    assert result["image_path"] is None


def test_build_vlm_text():
    builder = ExperimentInputBuilder()

    prompt = "Test benchmark prompt"

    result = builder.build_vlm_text(prompt)

    assert result["prompt"] == prompt
    assert result["image_path"] is None


def test_build_vlm_image(tmp_path: Path):
    builder = ExperimentInputBuilder()

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake image")

    result = builder.build_vlm_image(
        neutral_intro_prompt="Please examine the image.",
        image_path=image_path,
    )

    assert result.prompt == "Please examine the image."
    assert result.image_path == image_path


def test_build_vlm_image_rejects_empty_intro(tmp_path: Path):
    builder = ExperimentInputBuilder()

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake image")

    with pytest.raises(ValueError):
        builder.build_vlm_image(
            neutral_intro_prompt="",
            image_path=image_path,
        )


def test_build_vlm_image_rejects_missing_image(tmp_path: Path):
    builder = ExperimentInputBuilder()

    image_path = tmp_path / "does_not_exist.png"

    with pytest.raises(FileNotFoundError):
        builder.build_vlm_image(
            neutral_intro_prompt="Please examine the image.",
            image_path=image_path,
        )