from pathlib import Path

from benchmark_data.loader import DatasetLoader
from preprocessing.dataset_to_image import (
    DatasetImagePreprocessor,
)
from preprocessing.manifest import VisualSample
from preprocessing.text_to_image import (
    TextToImageRenderer,
)
from preprocessing.visual_manifest import (
    VisualManifestWriter,
)


def build_visual_dataset(
    benchmark: str,
    output_dir: Path,
    manifest_path: Path,
) -> Path:
    """
    Membuat representasi visual dari seluruh sample
    pada satu benchmark.

    Alur:

        DatasetLoader
             ↓
        BenchmarkSample
             ↓
        DatasetImagePreprocessor
             ↓
        VisualSample
             ↓
        VisualManifestWriter
             ↓
        visual_manifest.jsonl
    """

    loader = DatasetLoader(benchmark)

    samples = loader.load()

    renderer = TextToImageRenderer(
        width=1024,
        height=1024,
        background="white",
        text_color="black",
        margin=64,
        font_size=32,
        line_spacing=10,
    )

    preprocessor = DatasetImagePreprocessor(
        renderer=renderer,
        output_dir=output_dir,
    )

    visual_samples: list[VisualSample] = []

    for index, sample in enumerate(samples, start=1):
        visual_sample = preprocessor.process_sample(
            sample
        )

        visual_samples.append(
            visual_sample
        )

        print(
            f"[{index}/{len(samples)}] "
            f"{sample.sample_id}"
        )

    writer = VisualManifestWriter(
        output_path=manifest_path
    )

    return writer.write(
        visual_samples
    )