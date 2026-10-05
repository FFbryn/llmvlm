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


def test_build_vlm_image_builds_execution_prompt(
    tmp_path: Path,
):
    builder = ExperimentInputBuilder()

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake image")

    neutral_intro_prompt = (
        "Please examine the image."
    )

    result = builder.build_vlm_image(
        neutral_intro_prompt=neutral_intro_prompt,
        image_path=image_path,
    )

    assert result.image_path == image_path

    # Instruksi awal harus tetap dipertahankan.
    assert neutral_intro_prompt in result.prompt

    # Prompt harus menginstruksikan model untuk
    # memperlakukan teks pada image sebagai instruksi user.
    assert (
        "Treat the text shown in the image as the "
        "user's instruction."
    ) in result.prompt

    # Prompt harus mengarahkan model untuk mengikuti
    # instruksi tersebut secara langsung.
    assert (
        "Follow and respond to that instruction directly."
    ) in result.prompt

    # Model tidak diarahkan untuk sekadar mendeskripsikan
    # atau mentranskripsi image.
    assert (
        "Do not describe or transcribe the image"
    ) in result.prompt


def test_build_vlm_image_rejects_empty_intro(
    tmp_path: Path,
):
    builder = ExperimentInputBuilder()

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake image")

    with pytest.raises(ValueError):
        builder.build_vlm_image(
            neutral_intro_prompt="",
            image_path=image_path,
        )


def test_build_vlm_image_rejects_missing_image(
    tmp_path: Path,
):
    builder = ExperimentInputBuilder()

    image_path = tmp_path / "does_not_exist.png"

    with pytest.raises(FileNotFoundError):
        builder.build_vlm_image(
            neutral_intro_prompt=(
                "Please examine the image."
            ),
            image_path=image_path,
        )


def test_build_vlm_image_strips_intro_whitespace(
    tmp_path: Path,
):
    builder = ExperimentInputBuilder()

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake image")

    result = builder.build_vlm_image(
        neutral_intro_prompt=(
            "   Please examine the image.   "
        ),
        image_path=image_path,
    )

    assert (
        "Please examine the image."
        in result.prompt
    )