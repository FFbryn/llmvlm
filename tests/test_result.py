from pathlib import Path

import pytest

from config.model_config import GenerationConfig
from evaluation.result import ExperimentResult
from models.base import ModelResponse


def create_test_response():
    return ModelResponse(
        text="Test model response.",
        model_name="dummy-model",
        model_type="llm",
        sample_id="sample_001",
    )


def create_test_generation():
    return GenerationConfig(
        max_new_tokens=64,
        temperature=0.0,
        do_sample=False,
    )


def test_experiment_result_creation():
    result = ExperimentResult(
        sample_id="sample_001",
        benchmark="dummy_benchmark",
        behavior="dummy_behavior",
        category="dummy_category",
        model_name="dummy-model",
        model_type="llm",
        input_modality="text",
        prompt="Test prompt.",
        response="Test response.",
        max_new_tokens=64,
        temperature=0.0,
        do_sample=False,
    )

    assert result.sample_id == "sample_001"
    assert result.benchmark == "dummy_benchmark"
    assert result.behavior == "dummy_behavior"
    assert result.category == "dummy_category"
    assert result.model_name == "dummy-model"
    assert result.model_type == "llm"
    assert result.input_modality == "text"
    assert result.prompt == "Test prompt."
    assert result.response == "Test response."

    assert result.classification is None

    print(
        "[PASS] ExperimentResult creation"
    )


def test_experiment_result_from_model_response():
    response = create_test_response()
    generation = create_test_generation()

    result = ExperimentResult.from_model_response(
        sample_id="sample_001",
        benchmark="dummy_benchmark",
        prompt="Test prompt.",
        response=response,
        generation=generation,
        behavior="dummy_behavior",
        category="dummy_category",
        input_modality="text",
        device="cpu",
        torch_dtype="float32",
    )

    assert result.sample_id == "sample_001"
    assert result.benchmark == "dummy_benchmark"

    assert result.model_name == "dummy-model"
    assert result.model_type == "llm"

    assert result.input_modality == "text"

    assert result.prompt == "Test prompt."
    assert result.response == (
        "Test model response."
    )

    assert result.max_new_tokens == 64
    assert result.temperature == 0.0
    assert result.do_sample is False

    assert result.device == "cpu"
    assert result.torch_dtype == "float32"

    print(
        "[PASS] ExperimentResult.from_model_response"
    )


def test_experiment_result_with_classification():
    result = ExperimentResult(
        sample_id="sample_001",
        benchmark="dummy_benchmark",
        model_name="dummy-model",
        model_type="llm",
        input_modality="text",
        prompt="Test prompt.",
        response="Test response.",
        max_new_tokens=64,
    )

    classified = result.with_classification(
        "refusal"
    )

    assert result.classification is None

    assert classified.classification == (
        "refusal"
    )

    assert classified.sample_id == (
        result.sample_id
    )

    assert classified.response == (
        result.response
    )

    print(
        "[PASS] ExperimentResult classification"
    )


def test_experiment_result_to_dict():
    result = ExperimentResult(
        sample_id="sample_001",
        benchmark="dummy_benchmark",
        model_name="dummy-model",
        model_type="llm",
        input_modality="text",
        prompt="Test prompt.",
        response="Test response.",
        max_new_tokens=64,
        temperature=0.0,
        do_sample=False,
        device="cpu",
        torch_dtype="float32",
    )

    data = result.to_dict()

    assert isinstance(data, dict)

    assert data["sample_id"] == (
        "sample_001"
    )

    assert data["benchmark"] == (
        "dummy_benchmark"
    )

    assert data["model_name"] == (
        "dummy-model"
    )

    assert data["model_type"] == "llm"

    assert data["input_modality"] == "text"

    assert data["response"] == (
        "Test response."
    )

    assert data["max_new_tokens"] == 64

    print(
        "[PASS] ExperimentResult.to_dict"
    )


def test_experiment_result_image_path():
    image_path = Path(
        "tests/fixtures/vlm_test_image.png"
    )

    result = ExperimentResult(
        sample_id="sample_vlm_001",
        benchmark="dummy_benchmark",
        model_name="dummy-vlm",
        model_type="vlm",
        input_modality="image",
        prompt="Describe this image.",
        image_path=image_path,
        response="Test image response.",
        max_new_tokens=32,
    )

    data = result.to_dict()

    assert result.image_path == image_path

    assert data["image_path"] == str(
        image_path
    )

    assert result.model_type == "vlm"

    print(
        "[PASS] ExperimentResult image path"
    )


def test_experiment_result_rejects_empty_sample_id():
    with pytest.raises(ValueError):
        ExperimentResult(
            sample_id="",
            benchmark="dummy_benchmark",
            model_name="dummy-model",
            model_type="llm",
            input_modality="text",
        )

    print(
        "[PASS] Empty sample_id rejected"
    )


def test_experiment_result_rejects_empty_benchmark():
    with pytest.raises(ValueError):
        ExperimentResult(
            sample_id="sample_001",
            benchmark="",
            model_name="dummy-model",
            model_type="llm",
            input_modality="text",
        )

    print(
        "[PASS] Empty benchmark rejected"
    )


def test_experiment_result_rejects_empty_model_name():
    with pytest.raises(ValueError):
        ExperimentResult(
            sample_id="sample_001",
            benchmark="dummy_benchmark",
            model_name="",
            model_type="llm",
            input_modality="text",
        )

    print(
        "[PASS] Empty model_name rejected"
    )


def test_experiment_result_is_immutable():
    result = ExperimentResult(
        sample_id="sample_001",
        benchmark="dummy_benchmark",
        model_name="dummy-model",
        model_type="llm",
        input_modality="text",
    )

    with pytest.raises(AttributeError):
        result.sample_id = "changed"

    print(
        "[PASS] ExperimentResult immutability"
    )


def main():
    print("Running ExperimentResult tests...")
    print()

    test_experiment_result_creation()
    test_experiment_result_from_model_response()
    test_experiment_result_with_classification()
    test_experiment_result_to_dict()
    test_experiment_result_image_path()
    test_experiment_result_rejects_empty_sample_id()
    test_experiment_result_rejects_empty_benchmark()
    test_experiment_result_rejects_empty_model_name()
    test_experiment_result_is_immutable()

    print()
    print(
        "All ExperimentResult tests passed."
    )


if __name__ == "__main__":
    main()