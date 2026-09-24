from typing import Callable, Optional

from config.experiment_config import ExperimentConfig
from config.model_config_builder import ModelConfigBuilder
from evaluation.pipeline import ExperimentPipeline
from evaluation.three_condition_result import ThreeConditionResult
from evaluation.three_condition_runner import ThreeConditionRunner
from models.runner import ModelRunner
from preprocessing.manifest import VisualSample
from preprocessing.visual_manifest_loader import VisualManifestLoader


class ExperimentExecutor:
    """
    Orchestrator utama untuk menjalankan eksperimen
    tiga kondisi.

    Executor menghubungkan:

        Dataset sample
            ->
        Visual manifest
            ->
        ModelConfigBuilder
            ->
        ModelRunner
            ->
        ExperimentPipeline
            ->
        ThreeConditionRunner
            ->
        ThreeConditionResult

    Executor tidak melakukan:
        - classification
        - jailbreak detection
        - transferability analysis
        - statistical analysis
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        dataset_samples,
        model_runner_builder,
        llm_model_name: Optional[str] = None,
        vlm_model_name: Optional[str] = None,
        visual_manifest_loader: Optional[
            VisualManifestLoader
        ] = None,
        runner_factory: Optional[
            Callable[[ModelRunner], ModelRunner]
        ] = None,
    ):
        self.config = config
        self.dataset_samples = dataset_samples
        self.model_runner_builder = model_runner_builder

        self.llm_model_name = llm_model_name
        self.vlm_model_name = vlm_model_name

        self.visual_manifest_loader = (
            visual_manifest_loader
            if visual_manifest_loader is not None
            else VisualManifestLoader(
                config.visual_manifest_path
            )
        )

        self.runner_factory = runner_factory

    def _build_model_configs(self):
        """
        Membuat konfigurasi model untuk LLM dan VLM.
        """

        builder = ModelConfigBuilder(
            model_pair=self.config.model_pair,
            generation=self.config.generation,
            llm_model_name=self.llm_model_name,
            vlm_model_name=self.vlm_model_name,
            device=self.config.device,
            torch_dtype=self.config.torch_dtype,
        )

        llm_config = builder.build_llm()
        vlm_config = builder.build_vlm()

        return llm_config, vlm_config

    def _load_visual_samples(
        self,
    ) -> dict[str, VisualSample]:
        """
        Membaca visual manifest dan mengubahnya
        menjadi dictionary berdasarkan sample_id.
        """

        samples = self.visual_manifest_loader.load()

        return {
            sample.sample_id: sample
            for sample in samples
        }

    def _find_dataset_sample(self, sample_id: str):
        """
        Mencari sample asli berdasarkan sample_id.
        """

        for sample in self.dataset_samples:
            if sample.sample_id == sample_id:
                return sample

        raise ValueError(
            f"Dataset sample tidak ditemukan: {sample_id}"
        )

    def build_runner(self) -> ThreeConditionRunner:
        """
        Membuat ThreeConditionRunner.

        Pada tahap ini model belum dijalankan.
        """

        llm_config, vlm_config = self._build_model_configs()

        llm_runner = self.model_runner_builder.build(
            llm_config
        )

        vlm_runner = self.model_runner_builder.build(
            vlm_config
        )

        llm_pipeline = ExperimentPipeline(
            runner=llm_runner,
            generation=self.config.generation,
            benchmark=self.config.benchmark,
            device=self.config.device,
            torch_dtype=self.config.torch_dtype,
        )

        vlm_pipeline = ExperimentPipeline(
            runner=vlm_runner,
            generation=self.config.generation,
            benchmark=self.config.benchmark,
            device=self.config.device,
            torch_dtype=self.config.torch_dtype,
        )

        return ThreeConditionRunner(
            llm_pipeline=llm_pipeline,
            vlm_pipeline=vlm_pipeline,
        )

    def run_sample(
        self,
        sample_id: str,
    ) -> ThreeConditionResult:
        """
        Menjalankan satu sample pada tiga kondisi.
        """

        sample = self._find_dataset_sample(
            sample_id
        )

        visual_samples = self._load_visual_samples()

        if sample_id not in visual_samples:
            raise ValueError(
                "Visual sample tidak ditemukan untuk "
                f"sample_id={sample_id}"
            )

        visual_sample = visual_samples[sample_id]

        if visual_sample.benchmark != sample.benchmark:
            raise ValueError(
                "Benchmark dataset dan visual manifest "
                "tidak cocok untuk sample "
                f"{sample_id}."
            )

        runner = self.build_runner()

        return runner.run_sample(
            sample_id=sample.sample_id,
            prompt=sample.prompt,
            category=sample.category,
            visual_sample=visual_sample,
            neutral_intro_prompt=(
                self.config.neutral_intro_prompt
            ),
            metadata={
                "benchmark": sample.benchmark,
                "executor": "ExperimentExecutor",
            },
        )