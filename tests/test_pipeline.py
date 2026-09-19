from pathlib import Path

from config.model_config import GenerationConfig
from evaluation.pipeline import ExperimentPipeline
from evaluation.result import ExperimentResult
from models.base import BaseModel, ModelResponse
from models.runner import ModelRunner


class DummyModel(BaseModel):
    """
    Dummy model untuk integration test.

    Tidak menjalankan model sungguhan.
    """

    def __init__(
        self,
        model_name="dummy-model",
        model_type="llm",
    ):
        super().__init__(
            model_name=model_name,
            model_type=model_type,
        )

        self.loaded = False
        self.unloaded = False

    def load(self):
        self.loaded = True

    def generate(
        self,
        prompt,
        image_path=None,
        sample_id=None,
    ):
        return ModelResponse(
            text=f"Generated: {prompt}",
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
            image_path=image_path,
        )

    def unload(self):
        self.unloaded = True


def create_pipeline(
    model_type="llm",
):
    model = DummyModel(
        model_name="dummy-model",
        model_type=model_type,
    )

    runner = ModelRunner(model)

    generation = GenerationConfig(
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    )

    pipeline = ExperimentPipeline(
        runner=runner,
        generation=generation,
        benchmark="dummy_benchmark",
        device="cpu",
        torch_dtype="float32",
    )

    return pipeline, model


def test_pipeline_runs_one_sample():
    pipeline, model = create_pipeline()

    result = pipeline.run_sample(
        sample_id="sample_001",
        prompt="Explain what Python is.",
        behavior="dummy_behavior",
        category="dummy_category",
        input_modality="text",
    )

    assert isinstance(
        result,
        ExperimentResult,
    )

    assert result.sample_id == "sample_001"

    assert result.benchmark == (
        "dummy_benchmark"
    )

    assert result.behavior == (
        "dummy_behavior"
    )

    assert result.category == (
        "dummy_category"
    )

    assert result.model_name == (
        "dummy-model"
    )

    assert result.model_type == "llm"

    assert result.input_modality == "text"

    assert result.prompt == (
        "Explain what Python is."
    )

    assert result.response == (
        "Generated: Explain what Python is."
    )

    assert result.classification is None

    assert model.loaded is True
    assert model.unloaded is True


def test_pipeline_passes_metadata():
    pipeline, _ = create_pipeline()

    metadata = {
        "source": "dummy_dataset",
        "test_case": "integration",
    }

    result = pipeline.run_sample(
        sample_id="sample_002",
        prompt="Describe a computer.",
        metadata=metadata,
    )

    assert result.metadata == metadata


def test_pipeline_supports_vlm_image():
    image_path = Path(
        "tests/fixtures/vlm_test_image.png"
    )

    pipeline, model = create_pipeline(
        model_type="vlm"
    )

    result = pipeline.run_sample(
        sample_id="vlm_sample_001",
        prompt="Describe this image.",
        image_path=image_path,
        input_modality="image",
    )

    assert isinstance(
        result,
        ExperimentResult,
    )

    assert result.sample_id == (
        "vlm_sample_001"
    )

    assert result.model_type == "vlm"

    assert result.input_modality == "image"

    assert result.image_path == image_path

    assert result.response == (
        "Generated: Describe this image."
    )

    assert model.loaded is True
    assert model.unloaded is True


def test_pipeline_preserves_generation_config():
    pipeline, _ = create_pipeline()

    result = pipeline.run_sample(
        sample_id="sample_003",
        prompt="Test generation configuration.",
    )

    assert result.max_new_tokens == 32
    assert result.temperature == 0.0
    assert result.do_sample is False


def test_pipeline_preserves_runtime_metadata():
    pipeline, _ = create_pipeline()

    result = pipeline.run_sample(
        sample_id="sample_004",
        prompt="Test runtime metadata.",
    )

    assert result.device == "cpu"

    assert result.torch_dtype == (
        "float32"
    )


def main():
    print("Running ExperimentPipeline tests...")
    print()

    test_pipeline_runs_one_sample()
    test_pipeline_passes_metadata()
    test_pipeline_supports_vlm_image()
    test_pipeline_preserves_generation_config()
    test_pipeline_preserves_runtime_metadata()

    print()
    print(
        "All ExperimentPipeline tests passed."
    )


if __name__ == "__main__":
    main()