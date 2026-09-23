from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisualInput:
    """
    Represents the input used for the VLM image condition.
    """

    prompt: str
    image_path: Path


class ExperimentInputBuilder:
    """
    Builds model inputs for the experimental conditions.
    """

    def build_llm_text(self, original_prompt: str) -> dict:
        return {
            "prompt": original_prompt,
            "image_path": None,
        }

    def build_vlm_text(self, original_prompt: str) -> dict:
        return {
            "prompt": original_prompt,
            "image_path": None,
        }

    def build_vlm_image(
        self,
        *,
        neutral_intro_prompt: str,
        image_path: Path,
    ) -> VisualInput:
        if not neutral_intro_prompt.strip():
            raise ValueError(
                "neutral_intro_prompt tidak boleh kosong."
            )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image tidak ditemukan: {image_path}"
            )

        return VisualInput(
            prompt=neutral_intro_prompt,
            image_path=image_path,
        )