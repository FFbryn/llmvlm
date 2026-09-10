from pathlib import Path
from typing import Optional

from models.base import BaseModel, ModelResponse


class BaseLLM(BaseModel):
    """
    Abstract base class untuk seluruh Large Language Model (LLM)
    yang digunakan dalam project LLM vs LVLM.

    BaseLLM merupakan turunan dari BaseModel dan secara khusus
    mendefinisikan interface untuk model berbasis text.

    Contoh implementasi:
        - QwenLLM
        - VicunaLLM
    """

    def __init__(
        self,
        model_name: str,
    ):
        """
        Membuat object BaseLLM.

        Parameters
        ----------
        model_name : str
            Nama atau identifier model.
        """

        super().__init__(
            model_name=model_name,
            model_type="llm",
        )

    def generate(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Menghasilkan response dari LLM.

        LLM pada layer ini hanya menerima input text.
        Oleh karena itu image_path harus selalu None.

        Parameters
        ----------
        prompt : str
            Input text untuk LLM.

        image_path : Optional[Path]
            Path gambar.

            Untuk LLM parameter ini harus None.

        sample_id : Optional[str]
            ID sample benchmark.

        Returns
        -------
        ModelResponse
            Response dari LLM.

        Raises
        ------
        ValueError
            Jika image_path diberikan kepada LLM.
        """

        if image_path is not None:
            raise ValueError(
                "BaseLLM hanya menerima input text. "
                "image_path harus None."
            )

        return self._generate_text(
            prompt=prompt,
            sample_id=sample_id,
        )

    def _generate_text(
        self,
        prompt: str,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Interface internal untuk text generation.

        Implementasi konkret seperti QwenLLM dan VicunaLLM
        harus mengimplementasikan method ini.

        Parameters
        ----------
        prompt : str
            Input text.

        sample_id : Optional[str]
            ID sample benchmark.

        Returns
        -------
        ModelResponse
            Response model.
        """

        raise NotImplementedError(
            "Subclass BaseLLM harus mengimplementasikan "
            "_generate_text()."
        )