from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from config.model_config import GenerationConfig
from models.base import ModelResponse


@dataclass(frozen=True)
class ExperimentResult:
    """
    Struktur standar untuk menyimpan satu hasil eksperimen.

    ExperimentResult menyimpan hubungan antara:

        benchmark sample
            ↓
        model input
            ↓
        model response
            ↓
        classification

    Class ini belum menentukan apakah response merupakan
    jailbreak berhasil atau refusal. Classification akan
    ditentukan oleh evaluation layer.
    """

    # =========================
    # Sample information
    # =========================

    sample_id: str

    benchmark: str

    behavior: Optional[str] = None

    category: Optional[str] = None

    # =========================
    # Model information
    # =========================

    model_name: str = ""

    model_type: str = ""

    input_modality: str = ""

    # =========================
    # Input information
    # =========================

    prompt: str = ""

    image_path: Optional[Path] = None

    # =========================
    # Model output
    # =========================

    response: str = ""

    # =========================
    # Evaluation
    # =========================

    classification: Optional[str] = None

    # =========================
    # Generation configuration
    # =========================

    max_new_tokens: int = 0

    temperature: float = 0.0

    do_sample: bool = False

    # =========================
    # Runtime information
    # =========================

    device: Optional[str] = None

    torch_dtype: Optional[str] = None

    # =========================
    # Additional metadata
    # =========================

    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if not self.sample_id.strip():
            raise ValueError(
                "sample_id tidak boleh kosong."
            )

        if not self.benchmark.strip():
            raise ValueError(
                "benchmark tidak boleh kosong."
            )

        if not self.model_name.strip():
            raise ValueError(
                "model_name tidak boleh kosong."
            )

        if not self.model_type.strip():
            raise ValueError(
                "model_type tidak boleh kosong."
            )

        if not self.input_modality.strip():
            raise ValueError(
                "input_modality tidak boleh kosong."
            )

        if self.max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens tidak boleh negatif."
            )

    @classmethod
    def from_model_response(
        cls,
        *,
        sample_id: str,
        benchmark: str,
        prompt: str,
        response: ModelResponse,
        generation: GenerationConfig,
        behavior: Optional[str] = None,
        category: Optional[str] = None,
        input_modality: str = "text",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ExperimentResult":
        """
        Membuat ExperimentResult dari ModelResponse.

        Ini menjadi jembatan antara model layer dan
        evaluation/result layer.
        """

        return cls(
            sample_id=sample_id,
            benchmark=benchmark,
            behavior=behavior,
            category=category,
            model_name=response.model_name,
            model_type=response.model_type,
            input_modality=input_modality,
            prompt=prompt,
            image_path=response.image_path,
            response=response.text,
            max_new_tokens=(
                generation.max_new_tokens
            ),
            temperature=(
                generation.temperature
            ),
            do_sample=(
                generation.do_sample
            ),
            device=device,
            torch_dtype=torch_dtype,
            metadata=metadata,
        )

    def with_classification(
        self,
        classification: str,
    ) -> "ExperimentResult":
        """
        Menghasilkan ExperimentResult baru dengan
        classification yang diberikan.

        Object asli tidak diubah karena dataclass
        menggunakan frozen=True.
        """

        return ExperimentResult(
            sample_id=self.sample_id,
            benchmark=self.benchmark,
            behavior=self.behavior,
            category=self.category,
            model_name=self.model_name,
            model_type=self.model_type,
            input_modality=self.input_modality,
            prompt=self.prompt,
            image_path=self.image_path,
            response=self.response,
            classification=classification,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            device=self.device,
            torch_dtype=self.torch_dtype,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Mengubah ExperimentResult menjadi dictionary.

        Path dikonversi menjadi string agar lebih mudah
        disimpan sebagai JSON/CSV.
        """

        data = asdict(self)

        if self.image_path is not None:
            data["image_path"] = str(
                self.image_path
            )

        return data