from config.model_config import GenerationConfig
from config.model_config_builder import ModelConfigBuilder
from config.model_pair import ModelPairConfig
from evaluation.experiment_condition import ExperimentCondition


def create_test_builder() -> ModelConfigBuilder:
    pair = ModelPairConfig(
        pair_id="qwen_pair",
        llm_model_key="qwen_llm",
        vlm_model_key="qwen_vlm",
    )

    generation = GenerationConfig(
        max_new_tokens=64,
        temperature=0.0,
        do_sample=False,
    )

    return ModelConfigBuilder(
        model_pair=pair,
        generation=generation,
        llm_model_name="Qwen/Qwen2.5-7B-Instruct",
        vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device="cpu",
    )


def test_build_llm_config():
    builder = create_test_builder()

    config = builder.build_llm()

    assert config.model_key == "qwen_llm"
    assert (
        config.model_name
        == "Qwen/Qwen2.5-7B-Instruct"
    )
    assert config.device == "cpu"
    assert config.generation.max_new_tokens == 64


def test_build_vlm_config():
    builder = create_test_builder()

    config = builder.build_vlm()

    assert config.model_key == "qwen_vlm"
    assert (
        config.model_name
        == "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    assert config.device == "cpu"
    assert config.generation.max_new_tokens == 64


def test_llm_text_uses_llm_model():
    builder = create_test_builder()

    config = builder.build(
        ExperimentCondition.LLM_TEXT
    )

    assert config.model_key == "qwen_llm"
    assert (
        config.model_name
        == "Qwen/Qwen2.5-7B-Instruct"
    )


def test_vlm_text_uses_vlm_model():
    builder = create_test_builder()

    config = builder.build(
        ExperimentCondition.VLM_TEXT
    )

    assert config.model_key == "qwen_vlm"
    assert (
        config.model_name
        == "Qwen/Qwen2.5-VL-7B-Instruct"
    )


def test_vlm_image_uses_same_vlm_model():
    builder = create_test_builder()

    config = builder.build(
        ExperimentCondition.VLM_IMAGE
    )

    assert config.model_key == "qwen_vlm"
    assert (
        config.model_name
        == "Qwen/Qwen2.5-VL-7B-Instruct"
    )


def test_vlm_text_and_vlm_image_have_same_model_config():
    builder = create_test_builder()

    text_config = builder.build(
        ExperimentCondition.VLM_TEXT
    )

    image_config = builder.build(
        ExperimentCondition.VLM_IMAGE
    )

    assert text_config == image_config