from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from models.base import ModelResponse
from models.vlm.base import BaseVLM


class QwenVLM(BaseVLM):
    """
    Adapter untuk Qwen2.5-VL-7B-Instruct.

    Adapter ini menghubungkan implementasi Qwen2.5-VL
    dengan abstraction BaseVLM yang digunakan oleh
    project LLM vs VLM.

    Mendukung:

        VLM_TEXT
            -> text-only input

        VLM_IMAGE
            -> image + text input
    """

    DEFAULT_MODEL_NAME = (
        "Qwen/Qwen2.5-VL-7B-Instruct"
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ):
        super().__init__(
            model_name=model_name,
        )

        self.device = (
            device
            if device is not None
            else self._detect_device()
        )

        self.torch_dtype = (
            torch_dtype
            if torch_dtype is not None
            else self._default_dtype()
        )

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.processor = None
        self.model = None

    @staticmethod
    def _detect_device() -> str:
        """
        Menentukan device yang digunakan model.
        """

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def _default_dtype(self) -> torch.dtype:
        """
        Menentukan default torch dtype berdasarkan device.
        """

        if self.device == "cuda":
            return torch.float16

        return torch.float32

    def load(self) -> None:
        """
        Memuat processor dan Qwen2.5-VL model.
        """

        if (
            self.model is not None
            and self.processor is not None
        ):
            return

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
        )

        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
            )
        )

        self.model.to(self.device)
        self.model.eval()

    def _move_inputs_to_device(self, inputs):
        """
        Memindahkan processor output ke device model.
        """

        if hasattr(inputs, "to"):
            return inputs.to(self.device)

        return {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

    def _build_generation_kwargs(self) -> dict:
        """
        Membuat parameter generation secara konsisten.

        Temperature hanya dikirim jika sampling aktif.
        """

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }

        if self.do_sample:
            generation_kwargs["temperature"] = (
                self.temperature
            )

        return generation_kwargs

    def _generate_text(
        self,
        prompt: str,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Generate menggunakan Qwen2.5-VL dengan input text-only.

        Ini digunakan untuk kondisi:

            VLM_TEXT
        """

        if (
            self.model is None
            or self.processor is None
        ):
            raise RuntimeError(
                "QwenVLM belum di-load. "
                "Panggil load() sebelum generate()."
            )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = self._move_inputs_to_device(inputs)

        generation_kwargs = (
            self._build_generation_kwargs()
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[
            0,
            input_length:
        ]

        response_text = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return ModelResponse(
            text=response_text,
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
            image_path=None,
        )

    def _generate_multimodal(
        self,
        prompt: str,
        image_path: Path,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Generate menggunakan Qwen2.5-VL dengan image + text.

        Ini digunakan untuk kondisi:

            VLM_IMAGE
        """

        if (
            self.model is None
            or self.processor is None
        ):
            raise RuntimeError(
                "QwenVLM belum di-load. "
                "Panggil load() sebelum generate()."
            )

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image tidak ditemukan: {image_path}"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = self._move_inputs_to_device(inputs)

        generation_kwargs = (
            self._build_generation_kwargs()
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[
            0,
            input_length:
        ]

        response_text = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return ModelResponse(
            text=response_text,
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
            image_path=image_path,
        )

    def unload(self) -> None:
        """
        Membebaskan model dan processor dari memory.
        """

        self.model = None
        self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()