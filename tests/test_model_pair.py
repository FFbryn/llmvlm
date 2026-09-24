from config.model_pair import ModelPairConfig
from evaluation.experiment_condition import ExperimentCondition


def create_pair() -> ModelPairConfig:
    return ModelPairConfig(
        pair_id="qwen_pair",
        llm_model_key="qwen_llm",
        vlm_model_key="qwen_vlm",
    )


def test_model_pair_creation():
    pair = create_pair()

    assert pair.pair_id == "qwen_pair"
    assert pair.llm_model_key == "qwen_llm"
    assert pair.vlm_model_key == "qwen_vlm"


def test_llm_text_uses_llm_model_key():
    pair = create_pair()

    assert (
        pair.model_key_for(
            ExperimentCondition.LLM_TEXT
        )
        == "qwen_llm"
    )


def test_vlm_text_uses_vlm_model_key():
    pair = create_pair()

    assert (
        pair.model_key_for(
            ExperimentCondition.VLM_TEXT
        )
        == "qwen_vlm"
    )


def test_vlm_image_uses_vlm_model_key():
    pair = create_pair()

    assert (
        pair.model_key_for(
            ExperimentCondition.VLM_IMAGE
        )
        == "qwen_vlm"
    )


def test_to_dict():
    pair = create_pair()

    assert pair.to_dict() == {
        "pair_id": "qwen_pair",
        "llm_model_key": "qwen_llm",
        "vlm_model_key": "qwen_vlm",
    }