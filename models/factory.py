from typing import Any

from config.model_config import ModelConfig
from models.base import BaseModel
from models.llm.qwen import QwenLLM
from models.llm.vicuna import VicunaLLM
from models.vlm.llava import LLaVAVLM
from models.vlm.qwen_v1 import QwenVLM


MODEL_REGISTRY = {
    "qwen_llm": QwenLLM,
    "vicuna_llm": VicunaLLM,
    "qwen_vlm": QwenVLM,
    "llava_vlm": LLaVAVLM,
}


def create_model(
    model_key: str,
    **kwargs: Any,
) -> BaseModel:
    """
    Membuat instance model berdasarkan model key.

    Function ini hanya bertanggung jawab untuk
    membuat object model.

    Function ini tidak melakukan:
        - load model
        - generate response
        - unload model
        - membaca benchmark
        - menjalankan eksperimen
    """

    if model_key not in MODEL_REGISTRY:
        available_models = ", ".join(
            sorted(MODEL_REGISTRY.keys())
        )

        raise ValueError(
            f"Unknown model key: '{model_key}'. "
            f"Available models: {available_models}"
        )

    model_class = MODEL_REGISTRY[model_key]

    model = model_class(**kwargs)

    if not isinstance(model, BaseModel):
        raise TypeError(
            f"Model '{model_key}' tidak mengikuti "
            "BaseModel interface."
        )

    return model


def create_model_from_config(
    config: ModelConfig,
) -> BaseModel:
    """
    Membuat instance model berdasarkan ModelConfig.

    Function ini menerjemahkan konfigurasi eksperimen
    menjadi constructor arguments untuk model.

    Model tidak akan di-load di sini.

    Lifecycle tetap:

        create_model_from_config()
                ↓
             load()
                ↓
            generate()
                ↓
            unload()
    """

    if not isinstance(config, ModelConfig):
        raise TypeError(
            "config harus merupakan instance ModelConfig."
        )

    kwargs = {}

    if config.model_name is not None:
        kwargs["model_name"] = config.model_name

    if config.device is not None:
        kwargs["device"] = config.device

    if config.torch_dtype is not None:
        raise ValueError(
            "torch_dtype saat ini belum didukung melalui "
            "string configuration. "
            "Gunakan create_model() langsung dengan "
            "torch.dtype."
        )

    kwargs["max_new_tokens"] = (
        config.generation.max_new_tokens
    )

    kwargs["temperature"] = (
        config.generation.temperature
    )

    kwargs["do_sample"] = (
        config.generation.do_sample
    )

    return create_model(
        config.model_key,
        **kwargs,
    )