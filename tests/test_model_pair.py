import pytest

from config.model_pair import ModelPairConfig


def test_model_pair_config():
    config = ModelPairConfig(
        pair_id="qwen_pair",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    assert config.pair_id == "qwen_pair"
    assert config.llm_model == "Qwen/Qwen2.5-7B-Instruct"
    assert config.vlm_model == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_model_pair_to_dict():
    config = ModelPairConfig(
        pair_id="qwen_pair",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    assert config.to_dict() == {
        "pair_id": "qwen_pair",
        "llm_model": "Qwen/Qwen2.5-7B-Instruct",
        "vlm_model": "Qwen/Qwen2.5-VL-7B-Instruct",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("pair_id", ""),
        ("llm_model", ""),
        ("vlm_model", ""),
    ],
)
def test_model_pair_rejects_empty_values(field, value):
    kwargs = {
        "pair_id": "test",
        "llm_model": "test-llm",
        "vlm_model": "test-vlm",
    }

    kwargs[field] = value

    with pytest.raises(ValueError):
        ModelPairConfig(**kwargs)