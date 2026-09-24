from config.model_config import (
    GenerationConfig,
    ModelConfig,
)
from evaluation.model_runner_builder import ModelRunnerBuilder
from models.base import BaseModel
from models.llm.qwen import QwenLLM
from models.runner import ModelRunner


def test_build_returns_model_runner():
    config = ModelConfig(
        model_key="qwen_llm",
        generation=GenerationConfig(
            max_new_tokens=16,
            temperature=0.0,
            do_sample=False,
        ),
    )

    builder = ModelRunnerBuilder()

    runner = builder.build(config)

    assert isinstance(runner, ModelRunner)
    assert isinstance(runner.model, BaseModel)
    assert isinstance(runner.model, QwenLLM)


def test_build_preserves_model_configuration():
    config = ModelConfig(
        model_key="qwen_llm",
        model_name="custom/qwen-model",
        device="cpu",
        generation=GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            do_sample=False,
        ),
    )

    builder = ModelRunnerBuilder()

    runner = builder.build(config)

    assert runner.model.model_name == "custom/qwen-model"
    assert runner.model.device == "cpu"
    assert runner.model.max_new_tokens == 32


def test_build_rejects_invalid_config():
    builder = ModelRunnerBuilder()

    try:
        builder.build("invalid")
    except TypeError:
        return

    raise AssertionError(
        "ModelRunnerBuilder harus menolak config yang bukan ModelConfig."
    )