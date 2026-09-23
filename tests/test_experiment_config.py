from pathlib import Path

import pytest

from config.experiment_config import ExperimentConfig
from config.model_pair import ModelPairConfig
from evaluation.config import GenerationConfig
from evaluation.experiment_condition import ExperimentCondition


def make_config(
    *,
    conditions=None,
    neutral_intro_prompt="Please examine the image.",
):
    model_pair = ModelPairConfig(
        pair_id="qwen_pair",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    generation = GenerationConfig(
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
    )

    return ExperimentConfig(
        model_pair=model_pair,
        benchmark="jbb",
        generation=generation,
        visual_manifest_path=Path(
            "benchmark_data/generated/visual_manifest_jbb.jsonl"
        ),
        neutral_intro_prompt=neutral_intro_prompt,
        conditions=(
            conditions
            if conditions is not None
            else (
                ExperimentCondition.LLM_TEXT,
                ExperimentCondition.VLM_TEXT,
                ExperimentCondition.VLM_IMAGE,
            )
        ),
    )


def test_experiment_config_contains_three_conditions():
    config = make_config()

    assert config.conditions == (
        ExperimentCondition.LLM_TEXT,
        ExperimentCondition.VLM_TEXT,
        ExperimentCondition.VLM_IMAGE,
    )


def test_experiment_config_uses_visual_condition():
    config = make_config()

    assert config.uses_visual_condition is True


def test_experiment_config_to_dict():
    config = make_config()

    result = config.to_dict()

    assert result["benchmark"] == "jbb"

    assert result["model_pair"]["pair_id"] == "qwen_pair"

    assert result["model_pair"]["llm_model"] == (
        "Qwen/Qwen2.5-7B-Instruct"
    )

    assert result["model_pair"]["vlm_model"] == (
        "Qwen/Qwen2.5-VL-7B-Instruct"
    )

    assert result["conditions"] == [
        "llm_text",
        "vlm_text",
        "vlm_image",
    ]

    assert result["neutral_intro_prompt"] == (
        "Please examine the image."
    )


def test_experiment_config_rejects_empty_benchmark():
    with pytest.raises(ValueError):
        model_pair = ModelPairConfig(
            pair_id="test",
            llm_model="test-llm",
            vlm_model="test-vlm",
        )

        ExperimentConfig(
            model_pair=model_pair,
            benchmark="",
            generation=GenerationConfig(
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
            ),
            visual_manifest_path=Path("test.jsonl"),
            neutral_intro_prompt="Please examine the image.",
        )


def test_experiment_config_rejects_empty_neutral_prompt():
    with pytest.raises(ValueError):
        make_config(neutral_intro_prompt="")


def test_experiment_config_rejects_duplicate_conditions():
    with pytest.raises(ValueError):
        make_config(
            conditions=(
                ExperimentCondition.LLM_TEXT,
                ExperimentCondition.LLM_TEXT,
            )
        )