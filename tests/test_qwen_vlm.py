from pathlib import Path

import pytest

from models.base import ModelResponse
from models.vlm.qwen_v1 import QwenVLM


FIXTURE_IMAGE = Path(
    "tests/fixtures/vlm_test_image.png"
)


@pytest.mark.slow
def test_qwen_vlm_load_generate_unload():
    if not FIXTURE_IMAGE.exists():
        pytest.skip(
            f"Fixture image tidak ditemukan: "
            f"{FIXTURE_IMAGE}"
        )

    model = QwenVLM(
        max_new_tokens=32,
        do_sample=False,
        temperature=0.0,
    )

    try:
        model.load()

        response = model.generate(
            prompt="Describe this image in one short sentence.",
            image_path=FIXTURE_IMAGE,
            sample_id="qwen_vlm_smoke_001",
        )

        assert isinstance(
            response,
            ModelResponse,
        )

        assert response.text.strip() != ""

        assert response.model_name == (
            "Qwen/Qwen2.5-VL-7B-Instruct"
        )

        assert response.model_type == "vlm"

        assert response.sample_id == (
            "qwen_vlm_smoke_001"
        )

        assert response.image_path == (
            FIXTURE_IMAGE
        )

    finally:
        model.unload()

    assert model.model is None
    assert model.processor is None