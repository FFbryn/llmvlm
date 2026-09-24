from typing import Optional

from evaluation.input_builder import ExperimentInputBuilder
from evaluation.pipeline import ExperimentPipeline
from evaluation.three_condition_result import ThreeConditionResult
from preprocessing.manifest import VisualSample


class ThreeConditionRunner:
    def __init__(
        self,
        llm_pipeline: ExperimentPipeline,
        vlm_pipeline: ExperimentPipeline,
        input_builder: Optional[
            ExperimentInputBuilder
        ] = None,
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

        # -------------------------------------------------
        # 1. LLM + TEXT
        # -------------------------------------------------

        llm_input = (
            self.input_builder.build_llm_text(
                prompt
            )
        )

        llm_metadata = {
            **(metadata or {}),
            "condition": "llm_text",
            "source_prompt": prompt,
        }

        llm_result = self.llm_pipeline.run_sample(
            sample_id=sample_id,
            prompt=llm_input["prompt"],
            category=category,
            input_modality="text",
            metadata=llm_metadata,
        )

        # -------------------------------------------------
        # 2. VLM + TEXT
        # -------------------------------------------------

        vlm_text_input = (
            self.input_builder.build_vlm_text(
                prompt
            )
        )

        vlm_text_metadata = {
            **(metadata or {}),
            "condition": "vlm_text",
            "source_prompt": prompt,
        }

        vlm_text_result = self.vlm_pipeline.run_sample(
            sample_id=sample_id,
            prompt=vlm_text_input["prompt"],
            category=category,
            input_modality="text",
            metadata=vlm_text_metadata,
        )

        # -------------------------------------------------
        # 3. Validate visual sample
        # -------------------------------------------------

        if visual_sample is None:
            raise ValueError(
                "visual_sample diperlukan untuk "
                "kondisi VLM_IMAGE."
            )

        if visual_sample.sample_id != sample_id:
            raise ValueError(
                "sample_id visual_sample tidak cocok "
                "dengan sample_id eksperimen: "
                f"{visual_sample.sample_id} != {sample_id}"
            )

        if visual_sample.source_prompt is not None:
            if visual_sample.source_prompt != prompt:
                raise ValueError(
                    "source_prompt pada visual_sample "
                    "tidak cocok dengan prompt eksperimen."
                )

        if neutral_intro_prompt is None:
            raise ValueError(
                "neutral_intro_prompt diperlukan untuk "
                "kondisi VLM_IMAGE."
            )

        # -------------------------------------------------
        # 4. VLM + IMAGE
        # -------------------------------------------------

        visual_input = (
            self.input_builder.build_vlm_image(
                neutral_intro_prompt=neutral_intro_prompt,
                image_path=visual_sample.image_path,
            )
        )

        vlm_image_metadata = {
            **(metadata or {}),
            "condition": "vlm_image",
            "source_prompt": prompt,
            "neutral_intro_prompt": neutral_intro_prompt,
            "visual_sample_id": visual_sample.sample_id,
        }

        vlm_image_result = (
            self.vlm_pipeline.run_sample(
                sample_id=sample_id,
                prompt=visual_input.prompt,
                category=category,
                image_path=visual_input.image_path,
                input_modality="image",
                metadata=vlm_image_metadata,
            )
        )

        return ThreeConditionResult(
            sample_id=sample_id,
            llm_text=llm_result,
            vlm_text=vlm_text_result,
            vlm_image=vlm_image_result,
        )