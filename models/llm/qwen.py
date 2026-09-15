from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import ModelResponse
from models.llm.base import BaseLLM


class QwenLLM(BaseLLM):
    """
    Adapter untuk Qwen2.5-7B-Instruct.

    Class ini menghubungkan model Qwen dengan
    abstraction BaseLLM yang digunakan oleh
    project LLM vs LVLM.
    """

    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ):
        """
        Parameters
        ----------
        model_name : str
            Hugging Face model identifier atau
            path model lokal.

        device : Optional[str]
            Device yang digunakan model.

            Contoh:
                "cuda"
                "cpu"

            Jika None, device akan dipilih otomatis.

        torch_dtype : Optional[torch.dtype]
            Data type model.

            Jika None:
                CUDA -> torch.float16
                CPU  -> torch.float32

        max_new_tokens : int
            Jumlah maksimum token baru yang dihasilkan.

        temperature : float
            Temperature generation.

        do_sample : bool
            Apakah sampling digunakan.
        """

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
        Menentukan device yang tersedia.

        Returns
        -------
        str
            "cuda" jika CUDA tersedia,
            selain itu "cpu".
        """

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def _default_dtype(self) -> torch.dtype:
        """
        Menentukan dtype default berdasarkan device.
        """

        if self.device == "cuda":
            return torch.float16

        return torch.float32

    def load(self) -> None:
        """
        Memuat tokenizer dan model Qwen.

        Model tidak dimuat pada __init__ agar
        object dapat dibuat tanpa langsung
        menggunakan resource GPU/CPU besar.
        """

        if self.model is not None and self.tokenizer is not None:
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
        Menghasilkan response text menggunakan Qwen.

        Parameters
        ----------
        prompt : str
            Input text.

        sample_id : Optional[str]
            ID sample benchmark.

        Returns
        -------
        ModelResponse
            Response standar project.
        """

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "QwenLLM belum di-load. "
                "Panggil load() sebelum generate()."
            )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
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
        Membebaskan model dan tokenizer dari memory.
        """

        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()