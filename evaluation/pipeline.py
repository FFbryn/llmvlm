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
        Menjalankan satu sample.

        Method ini menggunakan runner yang sudah disediakan.
        """

        response = self.runner.run(
            prompt=prompt,
            image_path=image_path,
            sample_id=sample_id,
        )

        result = ExperimentResult.from_model_response(
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

        return result