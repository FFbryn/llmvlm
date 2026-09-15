from pathlib import Path

from models.base import BaseModel, ModelResponse
from models.vlm.base import BaseVLM


class DummyVLM(BaseVLM):
    def __init__(
        self,
        model_name: str = "dummy-vlm",
    ):
        super().__init__(
            model_name=model_name,
        )

        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def _generate_multimodal(
        self,
        prompt: str,
        image_path: Path | None = None,
        sample_id: str | None = None,
    ) -> ModelResponse:

        return ModelResponse(
            text=f"Dummy VLM response: {prompt}",
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
            image_path=image_path,
        )

    def unload(self) -> None:
        self.unloaded = True


def test_base_vlm_inherits_base_model():
    model = DummyVLM()

    assert isinstance(model, BaseModel)

    print("[PASS] BaseVLM inheritance test")


def test_base_vlm_initialization():
    model = DummyVLM()

    assert model.model_name == "dummy-vlm"
    assert model.model_type == "vlm"

    print("[PASS] BaseVLM initialization test")


def test_base_vlm_load_unload():
    model = DummyVLM()

    model.load()

    assert model.loaded is True

    model.unload()

    assert model.unloaded is True

    print("[PASS] BaseVLM load/unload test")


def test_base_vlm_generate_text_only():
    model = DummyVLM()

    response = model.generate(
        prompt="Describe an object.",
        sample_id="vlm_test_001",
    )

    assert isinstance(response, ModelResponse)

    assert response.text == (
        "Dummy VLM response: Describe an object."
    )

    assert response.model_name == "dummy-vlm"

    assert response.model_type == "vlm"

    assert response.sample_id == "vlm_test_001"

    assert response.image_path is None

    print("[PASS] BaseVLM text-only generation test")


def test_base_vlm_generate_with_image():
    model = DummyVLM()

    image_path = Path(
        "tests/fixtures/dummy_image.png"
    )

    response = model.generate(
        prompt="Describe the image.",
        image_path=image_path,
        sample_id="vlm_test_002",
    )

    assert isinstance(response, ModelResponse)

    assert response.model_type == "vlm"

    assert response.sample_id == "vlm_test_002"

    assert response.image_path == image_path

    print("[PASS] BaseVLM image generation test")


def main():
    print("Running BaseVLM tests...")
    print()

    test_base_vlm_inherits_base_model()
    test_base_vlm_initialization()
    test_base_vlm_load_unload()
    test_base_vlm_generate_text_only()
    test_base_vlm_generate_with_image()

    print()
    print("All BaseVLM tests passed.")


if __name__ == "__main__":
    main()