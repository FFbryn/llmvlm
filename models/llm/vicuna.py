from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import ModelResponse
from models.llm.base import BaseLLM


class VicunaLLM(BaseLLM):
    """
    Adapter untuk Vicuna-1.5-7B.

    Class ini menghubungkan model Vicuna dengan
    abstraction BaseLLM yang digunakan dalam
    project LLM vs LVLM.
    """

    DEFAULT_MODEL_NAME = "lmsys/vicuna-7b-v1.5"

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

        self.device = device or self._detect_device()

        self.torch_dtype = (
            torch_dtype
            if torch_dtype is not None
            else self._default_dtype()
        )

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.tokenizer = None
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
        Memuat tokenizer dan model Vicuna.
        """

        if (
            self.model is not None
            and self.tokenizer is not None
        ):
            return

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
        )

        self.model.to(self.device)

        self.model.eval()

    def _generate_text(
        self,
        prompt: str,
        sample_id: Optional[str] = None,
    ) -> ModelResponse:
        """
        Menghasilkan response text dari Vicuna.
        """

        if (
            self.model is None
            or self.tokenizer is None
        ):
            raise RuntimeError(
                "VicunaLLM belum di-load. "
                "Panggil load() sebelum generate()."
            )

        # Vicuna menggunakan format conversation sederhana.
        formatted_prompt = (
            "USER: "
            + prompt
            + "\nASSISTANT:"
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }

        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[
            0,
            input_length:,
        ]

        response_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return ModelResponse(
            text=response_text,
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
        )

    def unload(self) -> None:
        """
        Menghapus model dan tokenizer dari memory.
        """

        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()