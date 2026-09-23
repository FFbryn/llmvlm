from evaluation.result import ExperimentResult
from evaluation.three_condition_result import ThreeConditionResult


def make_result(
    *,
    sample_id: str,
    model_name: str,
    model_type: str,
    input_modality: str,
) -> ExperimentResult:
    return ExperimentResult(
        sample_id=sample_id,
        benchmark="test-benchmark",
        model_name=model_name,
        model_type=model_type,
        input_modality=input_modality,
        prompt="test prompt",
        response="test response",
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    )


def test_three_condition_result_accepts_matching_sample_ids():
    sample_id = "sample-001"

    llm_text = make_result(
        sample_id=sample_id,
        model_name="dummy-llm",
        model_type="llm",
        input_modality="text",
    )

    vlm_text = make_result(
        sample_id=sample_id,
        model_name="dummy-vlm",
        model_type="vlm",
        input_modality="text",
    )

    vlm_image = make_result(
        sample_id=sample_id,
        model_name="dummy-vlm",
        model_type="vlm",
        input_modality="image",
    )

    result = ThreeConditionResult(
        sample_id=sample_id,
        llm_text=llm_text,
        vlm_text=vlm_text,
        vlm_image=vlm_image,
    )

    assert result.sample_id == sample_id
    assert result.has_llm_text is True
    assert result.has_vlm_text is True
    assert result.has_vlm_image is True
    assert result.complete is True


def test_three_condition_result_allows_partial_results():
    sample_id = "sample-001"

    llm_text = make_result(
        sample_id=sample_id,
        model_name="dummy-llm",
        model_type="llm",
        input_modality="text",
    )

    result = ThreeConditionResult(
        sample_id=sample_id,
        llm_text=llm_text,
    )

    assert result.has_llm_text is True
    assert result.has_vlm_text is False
    assert result.has_vlm_image is False
    assert result.complete is False


def test_three_condition_result_rejects_mismatched_sample_id():
    llm_text = make_result(
        sample_id="sample-002",
        model_name="dummy-llm",
        model_type="llm",
        input_modality="text",
    )

    try:
        ThreeConditionResult(
            sample_id="sample-001",
            llm_text=llm_text,
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for mismatched sample_id."
    )