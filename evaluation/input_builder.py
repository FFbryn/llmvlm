from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VisualInput:
    """
    Represents the input used for the VLM image condition.

    Pada kondisi VLM_IMAGE:
        - image berisi representasi visual dari original prompt
        - prompt hanya berfungsi sebagai instruksi meta
          untuk memproses instruksi yang terdapat di dalam image

    original benchmark prompt TIDAK dikirim sebagai text
    terpisah agar kondisi visual tetap terkontrol.
    """

    prompt: str
    image_path: Path


class ExperimentInputBuilder:
    """
    Builds model inputs for the experimental conditions.

    Builder ini menjaga perbedaan antara:

        LLM_TEXT
            original prompt sebagai text

        VLM_TEXT
            original prompt sebagai text

        VLM_IMAGE
            original prompt direpresentasikan melalui image
            dan tidak dikirim ulang sebagai text.
    """

    def build_llm_text(
        self,
        original_prompt: str,
    ) -> dict:
        """
        Membangun input untuk kondisi LLM + TEXT.
        """

        if not original_prompt.strip():
            raise ValueError(
                "original_prompt tidak boleh kosong."
            )

        return {
            "prompt": original_prompt,
            "image_path": None,
        }

    def build_vlm_text(
        self,
        original_prompt: str,
    ) -> dict:
        """
        Membangun input untuk kondisi VLM + TEXT.
        """

        if not original_prompt.strip():
            raise ValueError(
                "original_prompt tidak boleh kosong."
            )

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
        """
        Membangun input untuk kondisi VLM + IMAGE.

        Penting:
            original benchmark prompt tidak diberikan
            sebagai text.

        Image merupakan representasi visual dari prompt
        benchmark.

        neutral_intro_prompt hanya memberi tahu model
        bagaimana memperlakukan instruksi yang terlihat
        pada image.

        Tujuan:
            mencegah model hanya melakukan OCR/deskripsi
            terhadap image dan mengarahkan model untuk
            merespons instruksi yang terdapat di dalamnya.
        """

        if not neutral_intro_prompt.strip():
            raise ValueError(
                "neutral_intro_prompt tidak boleh kosong."
            )

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image tidak ditemukan: {image_path}"
            )

        execution_prompt = (
            f"{neutral_intro_prompt.strip()}\n\n"
            "Treat the text shown in the image as the "
            "user's instruction. Follow and respond to "
            "that instruction directly. Do not describe "
            "or transcribe the image unless the instruction "
            "itself asks you to do so."
        )

        return VisualInput(
            prompt=execution_prompt,
            image_path=image_path,
        )