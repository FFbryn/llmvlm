from pathlib import Path
from typing import Optional

from models.base import BaseModel, ModelResponse


class ModelRunner:
    """
    Unified runner untuk menjalankan LLM dan VLM.

    ModelRunner bertanggung jawab terhadap lifecycle model:

        load -> generate -> unload

    Runner tidak mengetahui detail:
        - benchmark
        - dataset
        - klasifikasi response
        - transferability

    Runner hanya bertugas menjalankan model melalui
    interface BaseModel.
    """

    def __init__(self, model: BaseModel):
        if not isinstance(model, BaseModel):
            raise TypeError(
                "ModelRunner membutuhkan instance "
                "dari BaseModel."
            )

        self.model = model
        self.loaded = False

    def load(self) -> None:
        """
        Memuat model ke memory/device.
        """

        if self.loaded:
            return

        self.model.load()
        self.loaded = True

    def generate(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Menjalankan generation menggunakan model.

        Model harus sudah di-load sebelum generate().
        """

        if not self.loaded:
            raise RuntimeError(
                "Model belum di-load. "
                "Panggil load() sebelum generate()."
            )

        response = self.model.generate(
            prompt=prompt,
            image_path=image_path,
            sample_id=sample_id,
        )

        if not isinstance(response, ModelResponse):
            raise TypeError(
                "Model harus mengembalikan "
                "instance ModelResponse."
            )

        return response

    def unload(self) -> None:
        """
        Melepaskan model dari memory/device.
        """

        if not self.loaded:
            return

        self.model.unload()
        self.loaded = False

    def run(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Convenience method untuk menjalankan satu generation.

        Lifecycle:

            load
            generate
            unload

        Method ini cocok untuk eksekusi single sample.
        """

        self.load()

        try:
            return self.generate(
                prompt=prompt,
                image_path=image_path,
                sample_id=sample_id,
            )

        finally:
            self.unload()

    def __enter__(self):
        """
        Mendukung penggunaan:

            with ModelRunner(model) as runner:
                response = runner.generate(...)
        """

        self.load()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.unload()
        return False

    def __repr__(self) -> str:
        return (
            f"ModelRunner("
            f"model={self.model!r}, "
            f"loaded={self.loaded})"
        )