from pathlib import Path

from benchmark_data.loader import DatasetLoader
from preprocessing.dataset_to_image import DatasetImagePreprocessor
from preprocessing.text_to_image import TextToImageRenderer


def main():
    loader = DatasetLoader("jbb")

    samples = loader.load()

    print(f"Total samples tersedia: {len(samples)}")

    pilot_samples = samples[:5]

    renderer = TextToImageRenderer(
        width=1024,
        height=1024,
        margin=64,
        font_size=32,
        line_spacing=10,
    )

    preprocessor = DatasetImagePreprocessor(
        renderer=renderer,
        output_dir=Path(
            "benchmark_data/generated/text_rendered"
        ),
    )

    print(f"Pilot samples: {len(pilot_samples)}")
    print()

    for sample in pilot_samples:
        result = preprocessor.process_sample(sample)

        print(
            f"sample_id   : {result.sample_id}"
        )
        print(
            f"benchmark   : {result.benchmark}"
        )
        print(
            f"image_path  : {result.image_path}"
        )
        print("-" * 60)


if __name__ == "__main__":
    main()