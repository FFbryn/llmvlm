from enum import Enum


class ExperimentCondition(str, Enum):
    """
    Experimental conditions used in the LLM vs VLM study.
    """

    LLM_TEXT = "llm_text"
    VLM_TEXT = "vlm_text"
    VLM_IMAGE = "vlm_image"

    @property
    def model_type(self) -> str:
        if self is ExperimentCondition.LLM_TEXT:
            return "llm"

        return "vlm"

    @property
    def input_modality(self) -> str:
        if self in {
            ExperimentCondition.LLM_TEXT,
            ExperimentCondition.VLM_TEXT,
        }:
            return "text"

        return "image"

    @property
    def requires_image(self) -> bool:
        return self is ExperimentCondition.VLM_IMAGE
