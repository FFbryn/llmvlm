import pytest

from config.model_config import (
    GenerationConfig,
    ModelConfig,
)

from models.base import BaseModel
from models.factory import (
    create_model,
    create_model_from_config,
)

from models.model_registry import MODEL_REGISTRY

from models.llm.qwen import QwenLLM
from models.llm.vicuna import VicunaLLM
from models.vlm.llava import LLaVAVLM
from models.vlm.qwen_v1 import QwenVLM


def test_registry_contains_all_models():
    expected_models = {
        "qwen_llm",
        "vicuna_llm",
        "qwen_vlm",
        "llava_vlm",
    }

    assert expected_models.issubset(
        MODEL_REGISTRY.keys()
    )

    print(
        "[PASS] Model registry contains all models"
    )


@pytest.mark.parametrize(
    "model_key, expected_class",
    [
        ("qwen_llm", QwenLLM),
        ("vicuna_llm", VicunaLLM),
        ("qwen_vlm", QwenVLM),
        ("llava_vlm", LLaVAVLM),
    ],
)
def test_create_model_returns_correct_class(
    model_key,
    expected_class,
):
    model = create_model(model_key)

    assert isinstance(
        model,
        expected_class,
    )

    assert isinstance(
        model,
        BaseModel,
    )

    print(
        f"[PASS] Factory creates {model_key}"
    )


def test_create_model_rejects_unknown_model():
    with pytest.raises(ValueError) as error:
        create_model("unknown_model")

    assert "Unknown model key" in str(
        error.value
    )

    print(
        "[PASS] Factory rejects unknown model"
    )


def test_create_model_passes_kwargs():
    model = create_model(
        "qwen_llm",
        max_new_tokens=64,
        temperature=0.0,
        do_sample=False,
    )

    assert model.max_new_tokens == 64
    assert model.temperature == 0.0
    assert model.do_sample is False

    print(
        "[PASS] Factory passes constructor kwargs"
    )


def test_create_model_from_config():
    config = ModelConfig(
        model_key="qwen_llm",
        generation=GenerationConfig(
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False,
        ),
    )

    model = create_model_from_config(
        config
    )

    assert isinstance(
        model,
        QwenLLM,
    )

    assert isinstance(
        model,
        BaseModel,
    )

    assert model.max_new_tokens == 64
    assert model.temperature == 0.0
    assert model.do_sample is False

    print(
        "[PASS] Factory creates model from config"
    )


def test_create_model_from_config_with_model_name():
    config = ModelConfig(
        model_key="qwen_llm",
        model_name="custom/qwen-model",
        generation=GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            do_sample=False,
        ),
    )

    model = create_model_from_config(
        config
    )

    assert isinstance(
        model,
        QwenLLM,
    )

    assert model.model_name == (
        "custom/qwen-model"
    )

    assert model.max_new_tokens == 32

    print(
        "[PASS] Factory passes model_name from config"
    )


def test_create_model_from_config_with_device():
    config = ModelConfig(
        model_key="qwen_llm",
        device="cpu",
        generation=GenerationConfig(
            max_new_tokens=16,
            temperature=0.0,
            do_sample=False,
        ),
    )

    model = create_model_from_config(
        config
    )

    assert isinstance(
        model,
        QwenLLM,
    )

    assert model.device == "cpu"

    print(
        "[PASS] Factory passes device from config"
    )


def test_create_model_from_config_rejects_invalid_config():
    with pytest.raises(TypeError):
        create_model_from_config(
            "qwen_llm"
        )

    print(
        "[PASS] Factory rejects invalid config"
    )


def test_create_model_from_config_rejects_torch_dtype():
    config = ModelConfig(
        model_key="qwen_llm",
        torch_dtype="float16",
    )

    with pytest.raises(ValueError) as error:
        create_model_from_config(
            config
        )

    assert "torch_dtype" in str(
        error.value
    )

    print(
        "[PASS] Factory rejects unsupported torch_dtype"
    )


def test_create_all_models_from_config():
    model_configs = [
        (
            "qwen_llm",
            QwenLLM,
        ),
        (
            "vicuna_llm",
            VicunaLLM,
        ),
        (
            "qwen_vlm",
            QwenVLM,
        ),
        (
            "llava_vlm",
            LLaVAVLM,
        ),
    ]

    for model_key, expected_class in model_configs:
        config = ModelConfig(
            model_key=model_key,
            generation=GenerationConfig(
                max_new_tokens=16,
                temperature=0.0,
                do_sample=False,
            ),
        )

        model = create_model_from_config(
            config
        )

        assert isinstance(
            model,
            expected_class,
        )

        assert isinstance(
            model,
            BaseModel,
        )

    print(
        "[PASS] All models can be created from config"
    )


def main():
    print("Running Model Factory tests...")
    print()

    test_registry_contains_all_models()

    for model_key, expected_class in [
        ("qwen_llm", QwenLLM),
        ("vicuna_llm", VicunaLLM),
        ("qwen_vlm", QwenVLM),
        ("llava_vlm", LLaVAVLM),
    ]:
        test_create_model_returns_correct_class(
            model_key,
            expected_class,
        )

    test_create_model_rejects_unknown_model()
    test_create_model_passes_kwargs()
    test_create_model_from_config()
    test_create_model_from_config_with_model_name()
    test_create_model_from_config_with_device()
    test_create_model_from_config_rejects_invalid_config()
    test_create_model_from_config_rejects_torch_dtype()
    test_create_all_models_from_config()

    print()
    print(
        "All Model Factory tests passed."
    )


if __name__ == "__main__":
    main()