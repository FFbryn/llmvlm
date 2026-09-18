from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GenerationConfig:
    """
    Konfigurasi generation model.

    Parameter ini dibuat terpisah dari konfigurasi model
    agar generation setting dapat dicatat secara konsisten.
    """

    max_new_tokens: int = 256

    temperature: float = 0.0

    do_sample: bool = False

    def __post_init__(self):
        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens harus lebih besar dari 0."
            )

        if self.temperature < 0:
            raise ValueError(
                "temperature tidak boleh negatif."
            )

        if not self.do_sample and self.temperature != 0.0:
            raise ValueError(
                "Jika do_sample=False, "
                "temperature harus 0.0."
            )


@dataclass(frozen=True)
class ModelConfig:
    """
    Konfigurasi model yang digunakan dalam eksperimen.
    """

    model_key: str

    model_name: Optional[str] = None

    device: Optional[str] = None

    torch_dtype: Optional[str] = None

    generation: GenerationConfig = GenerationConfig()

    def __post_init__(self):
        if not self.model_key.strip():
            raise ValueError(
                "model_key tidak boleh kosong."
            )

    def to_dict(self) -> dict:
        """
        Mengubah konfigurasi menjadi dictionary.

        Berguna untuk logging dan reproducibility.
        """

        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "generation": {
                "max_new_tokens": (
                    self.generation.max_new_tokens
                ),
                "temperature": (
                    self.generation.temperature
                ),
                "do_sample": (
                    self.generation.do_sample
                ),
            },
        }