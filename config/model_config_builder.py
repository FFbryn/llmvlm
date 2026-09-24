from typing import Optional

from config.model_config import GenerationConfig, ModelConfig
from config.model_pair import ModelPairConfig
from evaluation.experiment_condition import ExperimentCondition


class ModelConfigBuilder:
    """
    Membangun ModelConfig berdasarkan ExperimentCondition.

    Builder ini tidak melakukan:
        - load model
        - inference
        - classification

    Tugasnya hanya:
        ExperimentCondition + ModelPairConfig
            -> ModelConfig
    """

    def __init__(
        self,
        model_pair: ModelPairConfig,
        generation: GenerationConfig,
        *,
        llm_model_name: Optional[str] = None,
        vlm_model_name: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ):
        self.model_pair = model_pair
        self.generation = generation
        self.llm_model_name = llm_model_name
        self.vlm_model_name = vlm_model_name
        self.device = device
        self.torch_dtype = torch_dtype

    def build(self, condition: ExperimentCondition) -> ModelConfig:
        model_key = self.model_pair.model_key_for(condition)

        if condition is ExperimentCondition.LLM_TEXT:
            model_name = self.llm_model_name
        else:
            model_name = self.vlm_model_name

        return ModelConfig(
            model_key=model_key,
            model_name=model_name,
            device=self.device,
            torch_dtype=self.torch_dtype,
            generation=self.generation,
        )

    def build_llm(self) -> ModelConfig:
        return self.build(ExperimentCondition.LLM_TEXT)

    def build_vlm(self) -> ModelConfig:
        return self.build(ExperimentCondition.VLM_TEXT)