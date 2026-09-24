from pathlib import Path

from config.model_config import GenerationConfig
from evaluation.pipeline import ExperimentPipeline
from evaluation.three_condition_runner import ThreeConditionRunner
from models.base import BaseModel, ModelResponse
from models.runner import ModelRunner
from preprocessing.manifest import VisualSample


class DummyLLM(BaseModel):
    def __init__(self):
        super().__init__(
            model_name="dummy-llm",
            model_type="llm",
        )

        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def generate(
        self,
        prompt: str,
        image_path: Path | None = None,
        sample_id: str | None = None,
    ) -> ModelResponse:
        return ModelResponse(
            text=f"LLM RESPONSE: {prompt}",
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
            image_path=image_path,
        )

    def unload(self) -> None:
        self.loaded = False


class DummyVLM(BaseModel):
    def __init__(self):
        super().__init__(
            model_name="dummy-vlm",
            model_type="vlm",
        )

        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def generate(
        self,
        prompt: str,
        image_path: Path | None = None,
        sample_id: str | None = None,
    ) -> ModelResponse:
        if image_path is None:
            text = f"VLM TEXT RESPONSE: {prompt}"
        else:
            text = (
                f"VLM IMAGE RESPONSE: "
                f"{prompt} | {image_path}"
            )

        return ModelResponse(
            text=text,
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
            image_path=image_path,
        )

    def unload(self) -> None:
        self.loaded = False


def create_test_pipelines():
    generation = GenerationConfig(
        max_new_tokens=16,
        temperature=0.0,
        do_sample=False,
    )

    llm_runner = ModelRunner(
        DummyLLM()
    )

    vlm_runner = ModelRunner(
        DummyVLM()
    )

    llm_pipeline = ExperimentPipeline(
        runner=llm_runner,
        generation=generation,
        benchmark="test",
        device="cpu",
    )

    vlm_pipeline = ExperimentPipeline(
        runner=vlm_runner,
        generation=generation,
        benchmark="test",
        device="cpu",
    )

    return llm_pipeline, vlm_pipeline


def test_three_condition_pipeline_end_to_end(
    tmp_path,
):
    image_path = tmp_path / "sample.png"

    image_path.write_bytes(
        b"dummy image content"
    )

    visual_sample = VisualSample(
        sample_id="sample-001",
        benchmark="test",
        image_path=image_path,
        source_prompt="original benchmark prompt",
        rendering_config={
            "width": 1024,
            "height": 1024,
        },
    )

    llm_pipeline, vlm_pipeline = (
        create_test_pipelines()
    )

    runner = ThreeConditionRunner(
        llm_pipeline=llm_pipeline,
        vlm_pipeline=vlm_pipeline,
    )

    result = runner.run_sample(
        sample_id="sample-001",
        prompt="original benchmark prompt",
        category="test-category",
        metadata={
            "test": True,
        },
        visual_sample=visual_sample,
        neutral_intro_prompt=(
            "Please respond to the provided input."
        ),
    )

    assert result.complete

    assert result.llm_text is not None
    assert result.vlm_text is not None
    assert result.vlm_image is not None

    assert (
        result.llm_text.response
        == "LLM RESPONSE: original benchmark prompt"
    )

    assert (
        result.vlm_text.response
        == "VLM TEXT RESPONSE: original benchmark prompt"
    )

    assert (
        "VLM IMAGE RESPONSE:"
        in result.vlm_image.response
    )

    assert (
        result.vlm_image.prompt
        == "Please respond to the provided input."
    )

    assert (
        result.vlm_image.metadata["source_prompt"]
        == "original benchmark prompt"
    )

    assert (
        result.vlm_image.metadata[
            "neutral_intro_prompt"
        ]
        == "Please respond to the provided input."
    )