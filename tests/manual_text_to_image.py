from pathlib import Path

from preprocessing.text_to_image import (
    TextToImageRenderer,
)


def main():
    renderer = TextToImageRenderer(
        width=1024,
        height=1024,
    )

    output_path = Path(
        "tests/fixtures/manual_text_render.png"
    )

    renderer.render(
        text=(
            "This is a manual test of the "
            "text-to-image preprocessing pipeline."
        ),
        output_path=output_path,
    )

    print(f"Image created: {output_path}")


if __name__ == "__main__":
    main()