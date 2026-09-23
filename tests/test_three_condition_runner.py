from pathlib import Path
from unittest.mock import MagicMock

from evaluation.config import GenerationConfig
from evaluation.pipeline import ExperimentPipeline
from evaluation.three_condition_runner import ThreeConditionRunner
from models.base import ModelResponse
from preprocessing.manifest import VisualSample


def make_pipeline(
    *,
    model_name: str,
    model_type: str,
) -> ExperimentPipeline:
    runner = MagicMock()

    def fake_run(
        *,
        prompt,
        image_path=None,
        sample_id=None,
    ):
        modality = "image" if image_path is not None else "text"

        return ModelResponse(
            text=f"{model_name} {modality} response",
            model_name=model_name,
            model_type=model_type,
        )

    runner.run.side_effect = fake_run

    return ExperimentPipeline(
        runner=runner,
        generation=GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            do_sample=False,
        ),
        benchmark="test-benchmark",
    )


def test_three_condition_runner():
    llm_pipeline = make_pipeline(
        model_name="dummy-llm",
        model_type="llm",
    )

    vlm_pipeline = make_pipeline(
        model_name="dummy-vlm",
        model_type="vlm",
    )

    runner = ThreeConditionRunner(
        llm_pipeline=llm_pipeline,
        vlm_pipeline=vlm_pipeline,
    )

    image_path = Path("tests/fixtures/vlm_test_image.png")

    visual_sample = VisualSample(
        sample_id="sample-001",
        benchmark="test-benchmark",
        image_path=image_path,
        source_prompt="Original benchmark prompt",
    )

    result = runner.run_sample(
        sample_id="sample-001",
        prompt="Original benchmark prompt",
        visual_sample=visual_sample,
        neutral_intro_prompt="Please examine the image.",
    )

    assert result.sample_id == "sample-001"

    # ----------------------------------------
    # Condition 1: LLM + Text
    # ----------------------------------------
    assert result.llm_text is not None
    assert result.llm_text.sample_id == "sample-001"
    assert result.llm_text.model_type == "llm"
    assert result.llm_text.input_modality == "text"

    # ----------------------------------------
    # Condition 2: VLM + Text
    # ----------------------------------------
    assert result.vlm_text is not None
    assert result.vlm_text.sample_id == "sample-001"
    assert result.vlm_text.model_type == "vlm"
    assert result.vlm_text.input_modality == "text"

    # ----------------------------------------
    # Condition 3: VLM + Image
    # ----------------------------------------
    assert result.vlm_image is not None
    assert result.vlm_image.sample_id == "sample-001"
    assert result.vlm_image.model_type == "vlm"
    assert result.vlm_image.input_modality == "image"
    assert result.vlm_image.image_path == image_path

    assert result.complete is True


def test_three_condition_runner_preserves_condition_metadata():
    llm_pipeline = make_pipeline(
        model_name="dummy-llm",
        model_type="llm",
    )

    vlm_pipeline = make_pipeline(
        model_name="dummy-vlm",
        model_type="vlm",
    )

    runner = ThreeConditionRunner(
        llm_pipeline=llm_pipeline,
        vlm_pipeline=vlm_pipeline,
    )

    image_path = Path("tests/fixtures/vlm_test_image.png")

    visual_sample = VisualSample(
        sample_id="sample-002",
        benchmark="test-benchmark",
        image_path=image_path,
        source_prompt="Original benchmark prompt",
    )

    result = runner.run_sample(
        sample_id="sample-002",
        prompt="Original benchmark prompt",
        visual_sample=visual_sample,
        neutral_intro_prompt="Please examine the image.",
        metadata={
            "source": "test",
        },
    )

    assert result.llm_text is not None
    assert result.vlm_text is not None
    assert result.vlm_image is not None

    assert result.llm_text.metadata["condition"] == "llm_text"
    assert result.vlm_text.metadata["condition"] == "vlm_text"
    assert result.vlm_image.metadata["condition"] == "vlm_image"

    assert result.vlm_image.metadata["source_prompt"] == (
        "Original benchmark prompt"
    )

    assert result.vlm_image.metadata["neutral_intro_prompt"] == (
        "Please examine the image."
    )

    assert result.llm_text.metadata["source"] == "test"
    assert result.vlm_text.metadata["source"] == "test"
    assert result.vlm_image.metadata["source"] == "test"


def test_three_condition_runner_requires_visual_sample():
    llm_pipeline = make_pipeline(
        model_name="dummy-llm",
        model_type="llm",
    )

    vlm_pipeline = make_pipeline(
        model_name="dummy-vlm",
        model_type="vlm",
    )

    runner = ThreeConditionRunner(
        llm_pipeline=llm_pipeline,
        vlm_pipeline=vlm_pipeline,
    )

    try:
        runner.run_sample(
            sample_id="sample-003",
            prompt="Original benchmark prompt",
            neutral_intro_prompt="Please examine the image.",
        )
    except ValueError as exc:
        assert "visual_sample" in str(exc)
        return

    raise AssertionError(
        "Expected ValueError when visual_sample is missing."
    )


def test_three_condition_runner_requires_neutral_intro():
    llm_pipeline = make_pipeline(
        model_name="dummy-llm",
        model_type="llm",
    )

    vlm_pipeline = make_pipeline(
        model_name="dummy-vlm",
        model_type="vlm",
    )

    runner = ThreeConditionRunner(
        llm_pipeline=llm_pipeline,
        vlm_pipeline=vlm_pipeline,
    )

    image_path = Path("tests/fixtures/vlm_test_image.png")

    visual_sample = VisualSample(
        sample_id="sample-004",
        benchmark="test-benchmark",
        image_path=image_path,
        source_prompt="Original benchmark prompt",
    )

    try:
        runner.run_sample(
            sample_id="sample-004",
            prompt="Original benchmark prompt",
            visual_sample=visual_sample,
        )
    except ValueError as exc:
        assert "neutral_intro_prompt" in str(exc)
        return

    raise AssertionError(
        "Expected ValueError when neutral_intro_prompt is missing."
    )