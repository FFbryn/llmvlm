import pytest

from models.base import ModelResponse
from models.llm.vicuna import VicunaLLM


@pytest.mark.slow
def test_vicuna_load_generate_unload():
    model = VicunaLLM(
        max_new_tokens=32,
        do_sample=False,
        temperature=0.0,
    )

    try:
        model.load()

        response = model.generate(
            prompt=(
                "Explain what machine learning is "
                "in one short sentence."
            ),
            sample_id="vicuna_smoke_001",
        )

        assert isinstance(response, ModelResponse)

        assert response.text.strip() != ""

        assert response.model_name == (
            "lmsys/vicuna-7b-v1.5"
        )

        assert response.model_type == "llm"

        assert response.sample_id == (
            "vicuna_smoke_001"
        )

        assert response.image_path is None

    finally:
        model.unload()

    assert model.model is None
    assert model.tokenizer is None