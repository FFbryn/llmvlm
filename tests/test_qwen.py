import pytest

from models.base import ModelResponse
from models.llm.qwen import QwenLLM


@pytest.mark.slow
def test_qwen_load_generate_unload():
    """
    Integration test untuk Qwen2.5-7B-Instruct.

    Test ini:
    1. Membuat QwenLLM.
    2. Memuat model.
    3. Menghasilkan response dari prompt benign.
    4. Memastikan response mengikuti ModelResponse.
    5. Membebaskan resource model.
    """

    model = QwenLLM(
        max_new_tokens=32,
        do_sample=False,
        temperature=0.0,
    )

    try:
        model.load()

        response = model.generate(
            prompt="Explain what a computer is in one short sentence.",
            sample_id="qwen_smoke_001",
        )

        assert isinstance(
            response,
            ModelResponse,
        )

        assert response.text.strip() != ""

        assert response.model_name == (
            "Qwen/Qwen2.5-7B-Instruct"
        )

        assert response.model_type == "llm"

        assert response.sample_id == (
            "qwen_smoke_001"
        )

        assert response.image_path is None

    finally:
        model.unload()

    assert model.model is None
    assert model.tokenizer is None