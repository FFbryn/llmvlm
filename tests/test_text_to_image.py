from pathlib import Path

from PIL import Image

from preprocessing.text_to_image import (
    TextToImageRenderer,
)


def test_text_to_image_creates_png(tmp_path):
    renderer = TextToImageRenderer(
        width=512,
        height=512,
    )

    output_path = tmp_path / "test.png"

    result = renderer.render(
        text="This is a test.",
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.is_file()


def test_text_to_image_has_expected_dimensions(tmp_path):
    renderer = TextToImageRenderer(
        width=640,
        height=480,
    )

    output_path = tmp_path / "dimensions.png"

    renderer.render(
        text="Dimension test.",
        output_path=output_path,
    )

    with Image.open(output_path) as image:
        assert image.size == (640, 480)
        assert image.format == "PNG"


def test_text_to_image_rejects_empty_text(tmp_path):
    renderer = TextToImageRenderer()

    output_path = tmp_path / "empty.png"

    try:
        renderer.render(
            text="",
            output_path=output_path,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_text_to_image_creates_parent_directory(tmp_path):
    renderer = TextToImageRenderer()

    output_path = (
        tmp_path
        / "nested"
        / "directory"
        / "test.png"
    )

    renderer.render(
        text="Nested directory test.",
        output_path=output_path,
    )

    assert output_path.exists()


def test_text_to_image_wraps_long_text(tmp_path):
    renderer = TextToImageRenderer(
        width=512,
        height=512,
        margin=32,
        font_size=32,
    )

    text = (
        "This is a long text that should be wrapped "
        "automatically instead of extending beyond "
        "the image boundaries."
    )

    output_path = tmp_path / "wrapped.png"

    result = renderer.render(
        text=text,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.is_file()

    with Image.open(output_path) as image:
        assert image.size == (512, 512)
        assert image.format == "PNG"


def test_text_to_image_rejects_text_that_does_not_fit(tmp_path):
    renderer = TextToImageRenderer(
        width=128,
        height=128,
        margin=16,
        font_size=32,
    )

    text = " ".join(
        ["This is a very long sentence."] * 100
    )

    output_path = tmp_path / "overflow.png"

    try:
        renderer.render(
            text=text,
            output_path=output_path,
        )
    except ValueError as exc:
        assert "tidak dapat dimuat" in str(exc)
    else:
        assert False, (
            "Expected ValueError karena teks "
            "melebihi kapasitas gambar."
        )