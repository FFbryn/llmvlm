from pathlib import Path

import pytest

from models.base import BaseModel, ModelResponse
from models.runner import ModelRunner


class DummyModel(BaseModel):
    """
    Model dummy untuk menguji ModelRunner
    tanpa menjalankan model sungguhan.
    """

    def __init__(
        self,
        model_name="dummy-model",
        model_type="llm",
    ):
        super().__init__(
            model_name=model_name,
            model_type=model_type,
        )

        self.load_count = 0
        self.unload_count = 0
        self.generate_count = 0

    def load(self) -> None:
        self.load_count += 1

    def generate(
        self,
        prompt: str,
        image_path: Path | None = None,
        sample_id: str | None = None,
    ) -> ModelResponse:

        self.generate_count += 1

        return ModelResponse(
            text=f"Dummy response: {prompt}",
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
            image_path=image_path,
        )

    def unload(self) -> None:
        self.unload_count += 1


def test_runner_accepts_base_model():
    model = DummyModel()

    runner = ModelRunner(model)

    assert runner.model is model
    assert runner.loaded is False

    print("[PASS] Runner accepts BaseModel")


def test_runner_rejects_invalid_model():
    with pytest.raises(TypeError):
        ModelRunner("not-a-model")

    print("[PASS] Runner rejects invalid model")


def test_runner_load():
    model = DummyModel()
    runner = ModelRunner(model)

    runner.load()

    assert runner.loaded is True
    assert model.load_count == 1

    print("[PASS] Runner load")


def test_runner_load_is_idempotent():
    model = DummyModel()
    runner = ModelRunner(model)

    runner.load()
    runner.load()
    runner.load()

    assert runner.loaded is True
    assert model.load_count == 1

    print("[PASS] Runner load is idempotent")


def test_runner_generate_requires_load():
    model = DummyModel()
    runner = ModelRunner(model)

    with pytest.raises(RuntimeError):
        runner.generate(
            prompt="test prompt",
            sample_id="runner_test_001",
        )

    print("[PASS] Runner requires load before generate")


def test_runner_generate():
    model = DummyModel()
    runner = ModelRunner(model)

    runner.load()

    response = runner.generate(
        prompt="Explain a computer.",
        sample_id="runner_test_001",
    )

    assert isinstance(response, ModelResponse)
    assert response.text == (
        "Dummy response: Explain a computer."
    )
    assert response.model_name == "dummy-model"
    assert response.model_type == "llm"
    assert response.sample_id == "runner_test_001"

    assert model.generate_count == 1

    print("[PASS] Runner generate")


def test_runner_unload():
    model = DummyModel()
    runner = ModelRunner(model)

    runner.load()
    runner.unload()

    assert runner.loaded is False
    assert model.unload_count == 1

    print("[PASS] Runner unload")


def test_runner_unload_is_idempotent():
    model = DummyModel()
    runner = ModelRunner(model)

    runner.load()

    runner.unload()
    runner.unload()
    runner.unload()

    assert runner.loaded is False
    assert model.unload_count == 1

    print("[PASS] Runner unload is idempotent")


def test_runner_run():
    model = DummyModel()
    runner = ModelRunner(model)

    response = runner.run(
        prompt="What is Python?",
        sample_id="runner_test_002",
    )

    assert isinstance(response, ModelResponse)
    assert response.text == (
        "Dummy response: What is Python?"
    )

    assert model.load_count == 1
    assert model.generate_count == 1
    assert model.unload_count == 1

    assert runner.loaded is False

    print("[PASS] Runner run")


def test_runner_run_unloads_after_error():
    class FailingModel(DummyModel):
        def generate(
            self,
            prompt,
            image_path=None,
            sample_id=None,
        ):
            self.generate_count += 1

            raise RuntimeError(
                "Simulated generation error"
            )

    model = FailingModel()
    runner = ModelRunner(model)

    with pytest.raises(RuntimeError):
        runner.run(
            prompt="Test failure handling.",
            sample_id="runner_error_001",
        )

    assert model.load_count == 1
    assert model.generate_count == 1
    assert model.unload_count == 1
    assert runner.loaded is False

    print("[PASS] Runner unloads after error")


def test_runner_context_manager():
    model = DummyModel()
    runner = ModelRunner(model)

    with runner as active_runner:
        assert active_runner is runner
        assert runner.loaded is True
        assert model.load_count == 1

        response = runner.generate(
            prompt="Describe Python.",
            sample_id="runner_context_001",
        )

        assert isinstance(response, ModelResponse)

    assert runner.loaded is False
    assert model.unload_count == 1

    print("[PASS] Runner context manager")


def test_runner_context_manager_unloads_after_error():
    model = DummyModel()
    runner = ModelRunner(model)

    with pytest.raises(RuntimeError):
        with runner:
            raise RuntimeError(
                "Simulated context error"
            )

    assert runner.loaded is False
    assert model.unload_count == 1

    print(
        "[PASS] Context manager unloads after error"
    )


def test_runner_supports_vlm_image():
    image_path = Path(
        "tests/fixtures/vlm_test_image.png"
    )

    model = DummyModel(
        model_name="dummy-vlm",
        model_type="vlm",
    )

    runner = ModelRunner(model)

    response = runner.run(
        prompt="Describe this image.",
        image_path=image_path,
        sample_id="runner_vlm_001",
    )

    assert isinstance(response, ModelResponse)
    assert response.model_type == "vlm"
    assert response.sample_id == "runner_vlm_001"
    assert response.image_path == image_path

    print("[PASS] Runner supports VLM image")


def main():
    print("Running ModelRunner tests...")
    print()

    test_runner_accepts_base_model()
    test_runner_rejects_invalid_model()
    test_runner_load()
    test_runner_load_is_idempotent()
    test_runner_generate_requires_load()
    test_runner_generate()
    test_runner_unload()
    test_runner_unload_is_idempotent()
    test_runner_run()
    test_runner_run_unloads_after_error()
    test_runner_context_manager()
    test_runner_context_manager_unloads_after_error()
    test_runner_supports_vlm_image()

    print()
    print("All ModelRunner tests passed.")


if __name__ == "__main__":
    main()