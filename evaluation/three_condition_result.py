from dataclasses import dataclass
from typing import Optional

from evaluation.result import ExperimentResult


@dataclass(frozen=True)
class ThreeConditionResult:
    """
    Results for one benchmark sample evaluated under the
    three experimental conditions.
    """

    sample_id: str

    llm_text: Optional[ExperimentResult] = None
    vlm_text: Optional[ExperimentResult] = None
    vlm_image: Optional[ExperimentResult] = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id tidak boleh kosong.")

        results = {
            "llm_text": self.llm_text,
            "vlm_text": self.vlm_text,
            "vlm_image": self.vlm_image,
        }

        for condition_name, result in results.items():
            if result is not None and result.sample_id != self.sample_id:
                raise ValueError(
                    f"sample_id pada {condition_name} tidak cocok "
                    f"dengan sample_id utama."
                )

    @property
    def has_llm_text(self) -> bool:
        return self.llm_text is not None

    @property
    def has_vlm_text(self) -> bool:
        return self.vlm_text is not None

    @property
    def has_vlm_image(self) -> bool:
        return self.vlm_image is not None

    @property
    def complete(self) -> bool:
        return (
            self.llm_text is not None
            and self.vlm_text is not None
            and self.vlm_image is not None
        )