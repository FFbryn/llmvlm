import pytest

from config.model_config import (
    GenerationConfig,
    ModelConfig,
)


def test_generation_config_defaults():
    config = GenerationConfig()

    assert config.max_new_tokens == 256
    assert config.temperature == 0.0
    assert config.do_sample is False

    print("[PASS] GenerationConfig defaults")


def test_generation_config_custom_values():
    config = GenerationConfig(
        max_new_tokens=64,
        temperature=0.7,
        do_sample=True,
    )

    assert config.max_new_tokens == 64
    assert config.temperature == 0.7
    assert config.do_sample is True

    print("[PASS] GenerationConfig custom values")


def test_generation_config_rejects_invalid_max_tokens():
    with pytest.raises(ValueError):
        GenerationConfig(
            max_new_tokens=0,
        )

    print("[PASS] Invalid max_new_tokens rejected")


def test_generation_config_rejects_negative_temperature():
    with pytest.raises(ValueError):
        GenerationConfig(
            temperature=-1.0,
        )

    print("[PASS] Negative temperature rejected")


def test_generation_config_rejects_inconsistent_sampling():
    with pytest.raises(ValueError):
        GenerationConfig(
            temperature=0.7,
            do_sample=False,
        )

    print(
        "[PASS] Inconsistent sampling configuration rejected"
    )


def test_model_config():
    generation = GenerationConfig(
        max_new_tokens=64,
        temperature=0.0,
        do_sample=False,
    )

    config = ModelConfig(
        model_key="qwen_llm",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="cuda",
        torch_dtype="float16",
        generation=generation,
    )

    assert config.model_key == "qwen_llm"
    assert config.model_name == (
        "Qwen/Qwen2.5-7B-Instruct"
    )
    assert config.device == "cuda"
    assert config.torch_dtype == "float16"

    assert config.generation.max_new_tokens == 64
    assert config.generation.temperature == 0.0
    assert config.generation.do_sample is False

    print("[PASS] ModelConfig")


def test_model_config_rejects_empty_model_key():
    with pytest.raises(ValueError):
        ModelConfig(
            model_key="",
        )

    print("[PASS] Empty model_key rejected")


def test_model_config_to_dict():
    config = ModelConfig(
        model_key="qwen_llm",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device="cuda",
        torch_dtype="float16",
        generation=GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            do_sample=False,
        ),
    )

    data = config.to_dict()

    assert data["model_key"] == "qwen_llm"
    assert data["model_name"] == (
        "Qwen/Qwen2.5-7B-Instruct"
    )
    assert data["device"] == "cuda"
    assert data["torch_dtype"] == "float16"

    assert data["generation"]["max_new_tokens"] == 32
    assert data["generation"]["temperature"] == 0.0
    assert data["generation"]["do_sample"] is False

    print("[PASS] ModelConfig.to_dict")


def test_model_config_is_immutable():
    config = ModelConfig(
        model_key="qwen_llm",
    )

    with pytest.raises(
        AttributeError
    ):
        config.model_key = "vicuna_llm"

    print("[PASS] ModelConfig immutability")


def main():
    print("Running ModelConfig tests...")
    print()

    test_generation_config_defaults()
    test_generation_config_custom_values()
    test_generation_config_rejects_invalid_max_tokens()
    test_generation_config_rejects_negative_temperature()
    test_generation_config_rejects_inconsistent_sampling()
    test_model_config()
    test_model_config_rejects_empty_model_key()
    test_model_config_to_dict()
    test_model_config_is_immutable()

    print()
    print("All ModelConfig tests passed.")


if __name__ == "__main__":
    main()