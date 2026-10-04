from dataclasses import dataclass

from evaluation.classification.labels import ClassificationLabel


@dataclass(frozen=True)
class ClassificationResult:
    """
    Hasil classification satu response.

    response_original menyimpan response asli tanpa perubahan.
    """

    response_original: str
    label: ClassificationLabel

    def __post_init__(self):
        if not isinstance(self.response_original, str):
            raise TypeError(
                "response_original harus berupa string."
            )

        if not isinstance(self.label, ClassificationLabel):
            raise TypeError(
                "label harus berupa ClassificationLabel."
            )

    @property
    def is_refusal(self) -> bool:
        return self.label == ClassificationLabel.REFUSAL

    @property
    def is_compliance(self) -> bool:
        return self.label == ClassificationLabel.COMPLIANCE

    @property
    def is_ambiguous(self) -> bool:
        return self.label == ClassificationLabel.AMBIGUOUS