from pathlib import Path
from typing import Optional

from config.model_config import GenerationConfig
from evaluation.result import ExperimentResult
from models.runner import ModelRunner


class ExperimentPipeline:
    """
    Pipeline dasar untuk menjalankan satu sample dataset
    melalui model dan menghasilkan ExperimentResult.

    Pipeline ini bertanggung jawab terhadap:

        DatasetSample
            ↓
        ModelRunner
            ↓
        ModelResponse
            ↓
        ExperimentResult

    Pipeline TIDAK bertanggung jawab terhadap:

        - response classification
        - jailbreak detection
        - transferability analysis
        - benchmark-specific scoring

    Lifecycle model:
        - run_sample() menggunakan lifecycle single-run:
              load -> generate -> unload

        - run_sample_loaded() mengasumsikan runner
          sudah di-load dan tidak melakukan load/unload.

          Method kedua digunakan untuk batch experiment.
    """

    def __init__(
        self,
        runner: ModelRunner,
        generation: GenerationConfig,
        benchmark: str,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ):
        self.runner = runner
        self.generation = generation
        self.benchmark = benchmark
        self.device = device
        self.torch_dtype = torch_dtype

    def _build_result(
        self,
        *,
        sample_id: str,
        prompt: str,
        response,
        behavior: Optional[str] = None,
        category: Optional[str] = None,
        image_path: Optional[Path] = None,
        input_modality: str = "text",
        metadata: Optional[dict] = None,
    ) -> ExperimentResult:
        """
        Mengubah ModelResponse menjadi ExperimentResult.

        Method ini tidak mengatur lifecycle model.
        """

        return ExperimentResult.from_model_response(
            sample_id=sample_id,
            benchmark=self.benchmark,
            prompt=prompt,
            response=response,
            generation=self.generation,
            behavior=behavior,
            category=category,
            input_modality=input_modality,
            device=self.device,
            torch_dtype=self.torch_dtype,
            metadata=metadata,
        )

    def run_sample(
        self,
        *,
        sample_id: str,
        prompt: str,
        behavior: Optional[str] = None,
        category: Optional[str] = None,
        image_path: Optional[Path] = None,
        input_modality: str = "text",
        metadata: Optional[dict] = None,
    ) -> ExperimentResult:
        """
        Menjalankan satu sample menggunakan lifecycle
        single-run.

        Lifecycle:

            load
            generate
            unload

        Method ini dipertahankan untuk:
            - single-sample smoke test
            - backward compatibility
            - eksekusi eksperimen individual
        """

        response = self.runner.run(
            prompt=prompt,
            image_path=image_path,
            sample_id=sample_id,
        )

        return self._build_result(
            sample_id=sample_id,
            prompt=prompt,
            response=response,
            behavior=behavior,
            category=category,
            image_path=image_path,
            input_modality=input_modality,
            metadata=metadata,
        )

    def run_sample_loaded(
        self,
        *,
        sample_id: str,
        prompt: str,
        behavior: Optional[str] = None,
        category: Optional[str] = None,
        image_path: Optional[Path] = None,
        input_modality: str = "text",
        metadata: Optional[dict] = None,
    ) -> ExperimentResult:
        """
        Menjalankan satu sample menggunakan runner
        yang SUDAH di-load.

        Lifecycle TIDAK dilakukan di method ini.

        Artinya method ini hanya:

            generate
            ↓
            ExperimentResult

        Method ini digunakan oleh batch experiment agar
        model tidak di-load dan di-unload untuk setiap sample.

        Caller bertanggung jawab memastikan:

            self.runner.load()

        sudah dilakukan sebelum method ini dipanggil.
        """

        if not self.runner.loaded:
            raise RuntimeError(
                "ModelRunner belum di-load. "
                "Gunakan runner.load() sebelum "
                "memanggil run_sample_loaded()."
            )

        response = self.runner.generate(
            prompt=prompt,
            image_path=image_path,
            sample_id=sample_id,
        )

        return self._build_result(
            sample_id=sample_id,
            prompt=prompt,
            response=response,
            behavior=behavior,
            category=category,
            image_path=image_path,
            input_modality=input_modality,
            metadata=metadata,
        )