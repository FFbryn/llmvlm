from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)

from models.base import ModelResponse
from models.vlm.base import BaseVLM


class LLaVAVLM(BaseVLM):
    """
    Adapter untuk LLaVA-1.5-7B.

    Adapter ini menghubungkan model LLaVA dengan
    abstraction BaseVLM yang digunakan dalam
    project LLM vs VLM.
    """

    DEFAULT_MODEL_NAME = (
        "llava-hf/llava-1.5-7b-hf"
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
        Menentukan default dtype berdasarkan device.
        """

        if self.device == "cuda":
            return torch.float16

        return torch.float32

    def load(self) -> None:
        """
        Memuat processor dan LLaVA model.
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
            LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
            )
        )

        self.model.to(self.device)
        self.model.eval()

    def _generate_multimodal(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Menghasilkan response dari LLaVA.

        Input:
            prompt      : text prompt
            image_path  : path menuju image
            sample_id   : ID sample eksperimen

        Output:
            ModelResponse
        """

        if (
            self.model is None
            or self.processor is None
        ):
            raise RuntimeError(
                "LLaVAVLM belum di-load. "
                "Panggil load() sebelum generate()."
            )

        if image_path is None:
            raise ValueError(
                "LLaVAVLM membutuhkan image_path "
                "untuk multimodal generation."
            )

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image tidak ditemukan: {image_path}"
            )

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image",
                    },
                ],
            },
        ]

        prompt_text = (
            self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )
        )

        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }

        if self.do_sample:
            generation_kwargs["temperature"] = (
                self.temperature
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
        Menghapus model dan processor dari memory.
        """

        self.model = None
        self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()