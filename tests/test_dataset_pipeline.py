from config.model_config import GenerationConfig
from benchmark_data.loader import DatasetLoader
from evaluation.pipeline import ExperimentPipeline
from evaluation.result import ExperimentResult
from models.base import BaseModel, ModelResponse
from models.runner import ModelRunner


class DummyModel(BaseModel):
    """
    Dummy model untuk integration test
    DatasetLoader -> ExperimentPipeline.

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

    def load(self):
        pass

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
        pass


def create_pipeline():
    model = DummyModel()

    runner = ModelRunner(model)

    generation = GenerationConfig(
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    )

    return ExperimentPipeline(
        runner=runner,
        generation=generation,
        benchmark="jbb",
        device="cpu",
        torch_dtype="float32",
    )


def test_dataset_to_pipeline_jbb():
    """
    Memastikan sample dari DatasetLoader
    dapat diteruskan ke ExperimentPipeline.
    """

    loader = DatasetLoader("jbb")
    samples = loader.load()

    assert len(samples) > 0

    sample = samples[0]

    pipeline = create_pipeline()

    result = pipeline.run_sample(
        sample_id=sample.sample_id,
        prompt=sample.prompt,
        category=sample.category,
        input_modality="text",
        metadata=sample.metadata,
    )

    assert isinstance(result, ExperimentResult)

    assert result.sample_id == sample.sample_id
    assert result.prompt == sample.prompt
    assert result.category == sample.category
    assert result.metadata == sample.metadata

    assert result.benchmark == "jbb"

    assert result.model_name == "dummy-model"
    assert result.model_type == "llm"

    assert result.input_modality == "text"

    assert result.response == (
        f"Generated: {sample.prompt}"
    )

    assert result.classification is None