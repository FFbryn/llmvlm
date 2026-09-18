from pathlib import Path
from typing import Optional

from models.base import BaseModel, ModelResponse


class BaseVLM(BaseModel):
    """
    Abstract base class untuk seluruh Vision-Language Model (VLM)
    yang digunakan dalam project LLM vs VLM.

    VLM dapat menerima input berupa:
        - text
        - image
        - text + image

    BaseVLM menyediakan interface umum agar implementasi
    model seperti Qwen-VL dan LLaVA dapat digunakan secara
    konsisten oleh pipeline eksperimen.
    """

    def __init__(
        self,
        model_name: str,
    ):
        super().__init__(
            model_name=model_name,
            model_type="vlm",
        )

    def generate(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Menghasilkan response dari VLM.

        image_path bersifat optional karena pada abstraction
        ini VLM secara teknis dapat menerima text-only maupun
        multimodal input.

        Implementasi spesifik model akan menangani bagaimana
        image_path diproses.
        """

        return self._generate_multimodal(
            prompt=prompt,
            image_path=image_path,
            sample_id=sample_id,
        )

    def _generate_multimodal(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Interface internal yang harus diimplementasikan
        oleh subclass VLM.
        """

        raise NotImplementedError(
            "Subclass BaseVLM harus mengimplementasikan "
            "_generate_multimodal()."
        )