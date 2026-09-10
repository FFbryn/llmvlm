from pathlib import Path

from models.base import ModelResponse
from models.llm.base import BaseLLM


class DummyLLM(BaseLLM):
    """
    Dummy LLM untuk menguji BaseLLM tanpa
    memuat model sungguhan.
    """

    def __init__(
        self,
        model_name: str = "dummy-llm",
    ):
        super().__init__(
            model_name=model_name,
        )

        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        """
        Simulasi proses loading model.
        """

        self.loaded = True

    def _generate_text(
        self,
        prompt: str,
        sample_id: str | None = None,
    ) -> ModelResponse:
        """
        Simulasi text generation.
        """

        return ModelResponse(
            text=f"Dummy response: {prompt}",
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
        )

    def unload(self) -> None:
        """
        Simulasi proses unloading model.
        """

        self.unloaded = True


def test_base_llm_initialization():
    """
    Memastikan BaseLLM mengatur model_type sebagai 'llm'.
    """

    model = DummyLLM()

    assert model.model_name == "dummy-llm"
    assert model.model_type == "llm"

    print("✓ BaseLLM initialization test berhasil.")


def test_base_llm_load_unload():
    """
    Memastikan load() dan unload() dapat digunakan.
    """

    model = DummyLLM()

    model.load()

    assert model.loaded is True

    model.unload()

    assert model.unloaded is True

    print("✓ BaseLLM load/unload test berhasil.")


def test_base_llm_generate():
    """
    Memastikan BaseLLM dapat menghasilkan ModelResponse.
    """

    model = DummyLLM()

    response = model.generate(
        prompt="test prompt",
        sample_id="test_001",
    )

    assert isinstance(
        response,
        ModelResponse,
    )

    assert response.text == (
        "Dummy response: test prompt"
    )

    assert response.model_name == "dummy-llm"
    assert response.model_type == "llm"
    assert response.sample_id == "test_001"
    assert response.image_path is None

    print("✓ BaseLLM generate test berhasil.")


def test_base_llm_rejects_image():
    """
    Memastikan LLM menolak input image.

    LLM layer hanya diperuntukkan bagi input text.
    """

    model = DummyLLM()

    try:
        model.generate(
            prompt="test prompt",
            image_path=Path(
                "dummy_image.png"
            ),
        )

    except ValueError as error:
        assert "image_path" in str(error)

        print(
            "✓ BaseLLM image rejection test berhasil."
        )

        return

    raise AssertionError(
        "BaseLLM seharusnya menolak image_path."
    )


def main():
    """
    Menjalankan seluruh test BaseLLM.
    """

    print(
        "Menjalankan BaseLLM tests..."
    )

    test_base_llm_initialization()
    test_base_llm_load_unload()
    test_base_llm_generate()
    test_base_llm_rejects_image()

    print()
    print(
        "Semua BaseLLM test berhasil."
    )


if __name__ == "__main__":
    main()