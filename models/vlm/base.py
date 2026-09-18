<<<<<<< HEAD
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelResponse:
    """
    Representasi standar response dari model.

    Class ini digunakan agar response dari LLM
    dan VLM memiliki struktur yang konsisten.
    """

    text: str

    model_name: str

    model_type: str

    sample_id: Optional[str] = None

    image_path: Optional[Path] = None

    metadata: Optional[dict] = None


class BaseModel(ABC):
    """
    Abstract base class untuk seluruh model
    yang digunakan dalam eksperimen.

    LLM dan VLM akan mengimplementasikan interface
    yang sama.

    Tujuan utama:

        model.generate(...)

    dapat digunakan secara konsisten oleh pipeline
    tanpa mengetahui detail implementasi model.
=======
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
>>>>>>> 3d6550e3787ddc6685889740a59fe8979d77a866
    """

    def __init__(
        self,
        model_name: str,
<<<<<<< HEAD
        model_type: str,
    ):
        """
        Parameters
        ----------
        model_name : str
            Nama model.

        model_type : str
            Jenis model.

            Contoh:
                - llm
                - vlm
        """

        self.model_name = model_name
        self.model_type = model_type

    @abstractmethod
    def load(self) -> None:
        """
        Memuat model dan komponen yang diperlukan.

        Method ini harus diimplementasikan oleh
        subclass.
        """

        raise NotImplementedError

    @abstractmethod
=======
    ):
        super().__init__(
            model_name=model_name,
            model_type="vlm",
        )

>>>>>>> 3d6550e3787ddc6685889740a59fe8979d77a866
    def generate(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
<<<<<<< HEAD
        Menghasilkan response dari model.

        Parameters
        ----------
        prompt : str
            Input text.

        image_path : Optional[Path]
            Path gambar jika model merupakan VLM.

        sample_id : Optional[str]
            ID sample benchmark.

        Returns
        -------
        ModelResponse
            Response standar model.
        """

        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """
        Membebaskan resource model dari memory.

        Method ini diperlukan terutama ketika
        beberapa model dijalankan dalam satu machine.
        """

        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Representasi object untuk debugging.
        """

        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"model_type='{self.model_type}')"
        )
=======
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
>>>>>>> 3d6550e3787ddc6685889740a59fe8979d77a866
