from pathlib import Path

from benchmark_data.schema import BenchmarkSample
from preprocessing.dataset_to_image import (
    DatasetImagePreprocessor,
)
from preprocessing.text_to_image import (
    TextToImageRenderer,
)


def test_dataset_sample_to_image(tmp_path):
    sample = BenchmarkSample(
        sample_id="test_001",
        benchmark="TEST",
        prompt="This is a dataset to image test.",
        category="test",
        metadata={
            "source": "unit_test",
        },
    )

    renderer = TextToImageRenderer(
        width=512,
        height=512,
    )

    preprocessor = DatasetImagePreprocessor(
        renderer=renderer,
        output_dir=tmp_path,
    )

    result = preprocessor.process_sample(
        sample
    )

    assert result.sample_id == sample.sample_id
    assert result.benchmark == sample.benchmark
    assert result.source_prompt == sample.prompt

    assert isinstance(
        result.image_path,
        Path,
    )

    assert result.image_path.exists()
    assert result.image_path.suffix == ".png"

    assert result.rendering_config is not None

    assert (
        result.rendering_config["width"]
        == 512
    )

    assert (
        result.rendering_config["height"]
        == 512
    )

    assert (
        result.rendering_config["margin"]
        == 64
    )