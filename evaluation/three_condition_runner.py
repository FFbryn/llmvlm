from typing import Optional

from evaluation.input_builder import ExperimentInputBuilder
from evaluation.pipeline import ExperimentPipeline
from evaluation.three_condition_result import ThreeConditionResult
from preprocessing.manifest import VisualSample


class ThreeConditionRunner:
    """
    Menjalankan tiga kondisi eksperimen:

        1. LLM + TEXT
        2. VLM + TEXT
        3. VLM + IMAGE

    Class ini tidak bertanggung jawab terhadap:

        - classification
        - jailbreak detection
        - transferability analysis
        - statistical analysis
        - model loading/unloading

    Lifecycle model untuk batch dikendalikan oleh caller.
    """

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

    def _validate_visual_sample(
        self,
        *,
        sample_id: str,
        prompt: str,
        visual_sample: Optional[VisualSample],
        neutral_intro_prompt: Optional[str],
    ) -> None:
        """
        Memvalidasi kebutuhan kondisi VLM_IMAGE.
        """

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
        """
        Menjalankan satu sample menggunakan lifecycle
        single-run.

        Setiap pipeline akan menggunakan runner.run(),
        sehingga model akan menjalankan:

            load -> generate -> unload
        """

        # -------------------------------------------------
        # 1. LLM + TEXT
        # -------------------------------------------------

        llm_input = self.input_builder.build_llm_text(
            prompt
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

        vlm_text_input = self.input_builder.build_vlm_text(
            prompt
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

        self._validate_visual_sample(
            sample_id=sample_id,
            prompt=prompt,
            visual_sample=visual_sample,
            neutral_intro_prompt=neutral_intro_prompt,
        )

        # -------------------------------------------------
        # 4. VLM + IMAGE
        # -------------------------------------------------

        visual_input = self.input_builder.build_vlm_image(
            neutral_intro_prompt=neutral_intro_prompt,
            image_path=visual_sample.image_path,
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

    def run_sample_loaded(
        self,
        *,
        sample_id: str,
        prompt: str,
        category: Optional[str] = None,
        metadata: Optional[dict] = None,
        visual_sample: Optional[VisualSample] = None,
        neutral_intro_prompt: Optional[str] = None,
    ) -> ThreeConditionResult:
        """
        Menjalankan satu sample ketika LLM dan VLM
        sudah di-load.

        Method ini TIDAK melakukan load/unload.

        Lifecycle harus dikendalikan oleh caller:

            llm_runner.load()
            vlm_runner.load()

            runner.run_sample_loaded(...)
            
            ...

            vlm_runner.unload()
            llm_runner.unload()
        """

        # -------------------------------------------------
        # Validasi runner
        # -------------------------------------------------

        if not self.llm_pipeline.runner.loaded:
            raise RuntimeError(
                "LLM ModelRunner belum di-load."
            )

        if not self.vlm_pipeline.runner.loaded:
            raise RuntimeError(
                "VLM ModelRunner belum di-load."
            )

        # -------------------------------------------------
        # 1. LLM + TEXT
        # -------------------------------------------------

        llm_input = self.input_builder.build_llm_text(
            prompt
        )

        llm_metadata = {
            **(metadata or {}),
            "condition": "llm_text",
            "source_prompt": prompt,
        }

        llm_result = self.llm_pipeline.run_sample_loaded(
            sample_id=sample_id,
            prompt=llm_input["prompt"],
            category=category,
            input_modality="text",
            metadata=llm_metadata,
        )

        # -------------------------------------------------
        # 2. VLM + TEXT
        # -------------------------------------------------

        vlm_text_input = self.input_builder.build_vlm_text(
            prompt
        )

        vlm_text_metadata = {
            **(metadata or {}),
            "condition": "vlm_text",
            "source_prompt": prompt,
        }

        vlm_text_result = (
            self.vlm_pipeline.run_sample_loaded(
                sample_id=sample_id,
                prompt=vlm_text_input["prompt"],
                category=category,
                input_modality="text",
                metadata=vlm_text_metadata,
            )
        )

        # -------------------------------------------------
        # 3. Validate visual sample
        # -------------------------------------------------

        self._validate_visual_sample(
            sample_id=sample_id,
            prompt=prompt,
            visual_sample=visual_sample,
            neutral_intro_prompt=neutral_intro_prompt,
        )

        # -------------------------------------------------
        # 4. VLM + IMAGE
        # -------------------------------------------------

        visual_input = self.input_builder.build_vlm_image(
            neutral_intro_prompt=neutral_intro_prompt,
            image_path=visual_sample.image_path,
        )

        vlm_image_metadata = {
            **(metadata or {}),
            "condition": "vlm_image",
            "source_prompt": prompt,
            "neutral_intro_prompt": neutral_intro_prompt,
            "visual_sample_id": visual_sample.sample_id,
        }

        vlm_image_result = (
            self.vlm_pipeline.run_sample_loaded(
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