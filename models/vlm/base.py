from pathlib import Path
from typing import Optional

from models.base import BaseModel, ModelResponse


class BaseVLM(BaseModel):
    """
    Abstract base class untuk seluruh Vision-Language Model (VLM)
    yang digunakan dalam project LLM vs VLM.

    VLM dapat menerima:

        1. Text-only input
        2. Image + text input

    Dalam project LLM vs VLM:

        image_path is None
            -> VLM_TEXT

        image_path is not None
            -> VLM_IMAGE

    Implementasi konkret:
        - QwenVLM
        - LLaVAVLM
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
        Generate response dari VLM.

        Jika image_path None:
            input dianggap VLM_TEXT.

        Jika image_path tersedia:
            input dianggap VLM_IMAGE.

        Parameters
        ----------
        prompt : str
            Prompt yang diberikan kepada VLM.

        image_path : Optional[Path]
            Path gambar jika menggunakan kondisi visual.

        sample_id : Optional[str]
            ID sample eksperimen.

        Returns
        -------
        ModelResponse
            Response standar project.
        """

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt VLM tidak boleh kosong."
            )

        if image_path is None:
            return self._generate_text(
                prompt=prompt,
                sample_id=sample_id,
            )

        return self._generate_multimodal(
            prompt=prompt,
            image_path=Path(image_path),
            sample_id=sample_id,
        )

    def _generate_text(
        self,
        prompt: str,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Interface internal untuk VLM text-only.

        Subclass konkret harus mengimplementasikan
        method ini jika mendukung kondisi VLM_TEXT.
        """

        raise NotImplementedError(
            "Subclass BaseVLM harus mengimplementasikan "
            "_generate_text() untuk kondisi VLM_TEXT."
        )

    def _generate_multimodal(
        self,
        prompt: str,
        image_path: Path,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Interface internal untuk VLM image + text.

        Subclass konkret harus mengimplementasikan
        method ini untuk kondisi VLM_IMAGE.
        """

        raise NotImplementedError(
            "Subclass BaseVLM harus mengimplementasikan "
            "_generate_multimodal() untuk kondisi VLM_IMAGE."
        )