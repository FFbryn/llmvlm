from typing import Optional

from evaluation.input_builder import ExperimentInputBuilder
from evaluation.pipeline import ExperimentPipeline
from evaluation.three_condition_result import ThreeConditionResult
from preprocessing.manifest import VisualSample


class ThreeConditionRunner:
    """
    Runs one benchmark sample under three experimental conditions:

    1. LLM + Text
    2. VLM + Text
    3. VLM + Rendered Image + Neutral Intro
    """

    def __init__(
        self,
        llm_pipeline: ExperimentPipeline,
        vlm_pipeline: ExperimentPipeline,
        input_builder: Optional[ExperimentInputBuilder] = None,
    ):
        self.llm_pipeline = llm_pipeline
        self.vlm_pipeline = vlm_pipeline
        self.input_builder = (
            input_builder
            if input_builder is not None
            else ExperimentInputBuilder()
        )

    def run_sample(
        self,
        *,
        sample_id: str,
        prompt: str,
        category: Optional[str] = None,
        metadata: Optional[dict] = None,
        visual_sample: Optional[VisualSample] = None,
        neutral_intro_prompt: Optional[str] = None,
    ) -> ThreeConditionResult:

        # --------------------------------------------------
        # Condition 1: LLM + Text
        # --------------------------------------------------
        llm_input = self.input_builder.build_llm_text(prompt)

        llm_result = self.llm_pipeline.run_sample(
            sample_id=sample_id,
            prompt=llm_input["prompt"],
            category=category,
            input_modality="text",
            metadata={
                **(metadata or {}),
                "condition": "llm_text",
            },
        )

        # --------------------------------------------------
        # Condition 2: VLM + Text
        # --------------------------------------------------
        vlm_text_input = self.input_builder.build_vlm_text(prompt)

        vlm_text_result = self.vlm_pipeline.run_sample(
            sample_id=sample_id,
            prompt=vlm_text_input["prompt"],
            category=category,
            input_modality="text",
            metadata={
                **(metadata or {}),
                "condition": "vlm_text",
            },
        )

        # --------------------------------------------------
        # Condition 3: VLM + Image
        # --------------------------------------------------
        if visual_sample is None:
            raise ValueError(
                "visual_sample diperlukan untuk kondisi VLM_IMAGE."
            )

        if neutral_intro_prompt is None:
            raise ValueError(
                "neutral_intro_prompt diperlukan untuk kondisi VLM_IMAGE."
            )

        visual_input = self.input_builder.build_vlm_image(
            neutral_intro_prompt=neutral_intro_prompt,
            image_path=visual_sample.image_path,
        )

        vlm_image_result = self.vlm_pipeline.run_sample(
            sample_id=sample_id,
            prompt=visual_input.prompt,
            category=category,
            image_path=visual_input.image_path,
            input_modality="image",
            metadata={
                **(metadata or {}),
                "condition": "vlm_image",
                "source_prompt": prompt,
                "neutral_intro_prompt": neutral_intro_prompt,
            },
        )

        return ThreeConditionResult(
            sample_id=sample_id,
            llm_text=llm_result,
            vlm_text=vlm_text_result,
            vlm_image=vlm_image_result,
        )