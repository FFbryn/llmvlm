from pathlib import Path

from models.base import BaseModel
from models.base import ModelResponse


class DummyModel(BaseModel):
    """
    Dummy model hanya untuk menguji interface BaseModel.

    Class ini bukan model AI sungguhan.
    """

    def __init__(self):
        super().__init__(
            model_name="dummy-model",
            model_type="llm",
        )

    def load(self) -> None:
        """
        Dummy load.
        """

        return None

    def generate(
        self,
        prompt: str,
        image_path=None,
        sample_id=None,
    ) -> ModelResponse:
        """
        Dummy generation.
        """

        return ModelResponse(
            text="dummy response",
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
            image_path=image_path,
            metadata={
                "dummy": True,
            },
        )

    def unload(self) -> None:
        """
        Dummy unload.
        """

        return None


def test_model_response():
    """
    Menguji ModelResponse.
    """

    response = ModelResponse(
        text="test response",
        model_name="test-model",
        model_type="llm",
        sample_id="sample_001",
    )

    assert response.text == "test response"

    assert (
        response.model_name
        == "test-model"
    )

    assert (
        response.model_type
        == "llm"
    )

    assert (
        response.sample_id
        == "sample_001"
    )

    assert response.image_path is None


def test_dummy_model():
    """
    Menguji implementasi BaseModel.
    """

    model = DummyModel()

    assert (
        model.model_name
        == "dummy-model"
    )

    assert (
        model.model_type
        == "llm"
    )

    response = model.generate(
        prompt="test prompt",
        sample_id="sample_001",
    )

    assert isinstance(
        response,
        ModelResponse,
    )

    assert (
        response.text
        == "dummy response"
    )

    assert (
        response.sample_id
        == "sample_001"
    )


def test_vlm_style_response():
    """
    Memastikan ModelResponse juga dapat menyimpan
    path gambar untuk kebutuhan VLM.
    """

    image_path = Path(
        "data/images/sample.png"
    )

    response = ModelResponse(
        text="test visual response",
        model_name="test-vlm",
        model_type="vlm",
        sample_id="sample_002",
        image_path=image_path,
    )

    assert (
        response.model_type
        == "vlm"
    )

    assert (
        response.image_path
        == image_path
    )


def main():
    print(
        "Menjalankan BaseModel tests..."
    )

    test_model_response()

    print(
        "✓ ModelResponse test berhasil."
    )

    test_dummy_model()

    print(
        "✓ DummyModel test berhasil."
    )

    test_vlm_style_response()

    print(
        "✓ VLM-style response test berhasil."
    )

    print(
        "\nSemua BaseModel test berhasil."
    )


if __name__ == "__main__":
    main()