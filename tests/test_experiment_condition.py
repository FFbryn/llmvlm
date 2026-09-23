from evaluation.experiment_condition import ExperimentCondition


def test_llm_text_condition():
    condition = ExperimentCondition.LLM_TEXT

    assert condition.value == "llm_text"
    assert condition.model_type == "llm"
    assert condition.input_modality == "text"
    assert condition.requires_image is False


def test_vlm_text_condition():
    condition = ExperimentCondition.VLM_TEXT

    assert condition.value == "vlm_text"
    assert condition.model_type == "vlm"
    assert condition.input_modality == "text"
    assert condition.requires_image is False


def test_vlm_image_condition():
    condition = ExperimentCondition.VLM_IMAGE

    assert condition.value == "vlm_image"
    assert condition.model_type == "vlm"
    assert condition.input_modality == "image"
    assert condition.requires_image is True
