from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.model_config import GenerationConfig
from config.model_pair import ModelPairConfig
from evaluation.experiment_condition import ExperimentCondition


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for one three-condition experiment.

    The configuration defines:
        - model pair
        - benchmark
        - generation settings
        - visual manifest
        - neutral introductory prompt

    It does not execute the experiment.
    """

    model_pair: ModelPairConfig
    benchmark: str
    generation: GenerationConfig
    visual_manifest_path: Path
    neutral_intro_prompt: str

    conditions: tuple[ExperimentCondition, ...] = (
        ExperimentCondition.LLM_TEXT,
        ExperimentCondition.VLM_TEXT,
        ExperimentCondition.VLM_IMAGE,
    )

    device: Optional[str] = None
    torch_dtype: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.benchmark.strip():
            raise ValueError("benchmark tidak boleh kosong.")

        if not self.neutral_intro_prompt.strip():
            raise ValueError(
                "neutral_intro_prompt tidak boleh kosong."
            )

        if not self.conditions:
            raise ValueError(
                "Minimal satu experimental condition harus tersedia."
            )

        unique_conditions = set(self.conditions)

        if len(unique_conditions) != len(self.conditions):
            raise ValueError(
                "Experimental conditions tidak boleh duplikat."
            )

    @property
    def uses_visual_condition(self) -> bool:
        return ExperimentCondition.VLM_IMAGE in self.conditions

    def to_dict(self) -> dict:
        return {
            "model_pair": self.model_pair.to_dict(),
            "benchmark": self.benchmark,
            "generation": {
                "max_new_tokens": self.generation.max_new_tokens,
                "temperature": self.generation.temperature,
                "do_sample": self.generation.do_sample,
            },
            "visual_manifest_path": str(
                self.visual_manifest_path
            ),
            "neutral_intro_prompt": self.neutral_intro_prompt,
            "conditions": [
                condition.value
                for condition in self.conditions
            ],
            "device": self.device,
            "torch_dtype": self.torch_dtype,
        }