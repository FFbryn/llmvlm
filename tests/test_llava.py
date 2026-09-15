import pytest

from models.base import ModelResponse
from models.vlm.llava import LLaVAVLM


FIXTURE_IMAGE = (
    "tests/fixtures/vlm_test_image.png"
)


@pytest.mark.slow
def test_llava_load_generate_unload():
    model = LLaVAVLM(
        max_new_tokens=32,
        do_sample=False,
        temperature=0.0,
    )

    try:
        model.load()

        response = model.generate(
            prompt=(
                "Describe this image "
                "in one short sentence."
            ),
            image_path=FIXTURE_IMAGE,
            sample_id="llava_smoke_001",
        )

        assert isinstance(
            response,
            ModelResponse,
        )

        assert response.text.strip() != ""

        assert response.model_name == (
            "llava-hf/llava-1.5-7b-hf"
        )

        assert response.model_type == "vlm"

        assert response.sample_id == (
            "llava_smoke_001"
        )

        assert response.image_path is not None

    finally:
        model.unload()

    assert model.model is None
    assert model.processor is None