from pathlib import Path

from benchmark_data.schema import BenchmarkSample
from config.model_config import GenerationConfig
from evaluation.pipeline import ExperimentPipeline
from models.base import BaseModel, ModelResponse
from models.runner import ModelRunner
from preprocessing.dataset_to_image import (
    DatasetImagePreprocessor,
)
from preprocessing.text_to_image import (
    TextToImageRenderer,
)


class DummyVLM(BaseModel):
    def __init__(self):
        super().__init__(
            model_name="dummy-vlm",
            model_type="vlm",
        )

    def load(self):
        pass

    def unload(self):
        pass

    def generate(
        self,
        prompt,
        image_path=None,
        sample_id=None,
    ):
        return ModelResponse(
            text="Dummy VLM response",
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
            image_path=image_path,
        )


def test_visual_sample_can_run_through_pipeline(
    tmp_path,
):
    sample = BenchmarkSample(
        sample_id="visual_001",
        benchmark="TEST",
        prompt="This is a visual pipeline test.",
        category="test",
        metadata={
            "source": "integration_test",
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

    visual_sample = (
        preprocessor.process_sample(sample)
    )

    runner = ModelRunner(
        DummyVLM()
    )

    generation = GenerationConfig(
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    )

    pipeline = ExperimentPipeline(
        runner=runner,
        generation=generation,
        benchmark=sample.benchmark,
    )

    result = pipeline.run_sample(
        sample_id=visual_sample.sample_id,
        prompt=visual_sample.source_prompt,
        category=sample.category,
        image_path=visual_sample.image_path,
        input_modality="image",
        metadata=sample.metadata,
    )

    assert result.sample_id == sample.sample_id
    assert result.benchmark == sample.benchmark

    assert result.input_modality == "image"

    assert (
        result.image_path
        == visual_sample.image_path
    )

    assert (
        result.response
        == "Dummy VLM response"
    )

    assert result.image_path.exists()