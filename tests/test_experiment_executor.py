from pathlib import Path

import pytest

from benchmark_data.schema import BenchmarkSample
from config.experiment_config import ExperimentConfig
from config.model_config import GenerationConfig
from config.model_pair import ModelPairConfig
from evaluation.experiment_executor import ExperimentExecutor
from evaluation.model_runner_builder import ModelRunnerBuilder
from models.base import BaseModel, ModelResponse
from models.runner import ModelRunner
from preprocessing.manifest import VisualSample
from preprocessing.visual_manifest_loader import VisualManifestLoader


class DummyLLM(BaseModel):
    def __init__(self):
        super().__init__(
            model_name="dummy-llm",
            model_type="llm",
        )
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def generate(
        self,
        prompt: str,
        image_path: Path | None = None,
        sample_id: str | None = None,
    ) -> ModelResponse:
        if not self.loaded:
            raise RuntimeError(
                "DummyLLM belum di-load."
            )

        return ModelResponse(
            text=f"LLM RESPONSE: {prompt}",
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
            image_path=image_path,
        )

    def unload(self) -> None:
        self.loaded = False


class DummyVLM(BaseModel):
    def __init__(self):
        super().__init__(
            model_name="dummy-vlm",
            model_type="vlm",
        )
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def generate(
        self,
        prompt: str,
        image_path: Path | None = None,
        sample_id: str | None = None,
    ) -> ModelResponse:
        if not self.loaded:
            raise RuntimeError(
                "DummyVLM belum di-load."
            )

        if image_path is not None:
            response_text = (
                f"VLM IMAGE RESPONSE: {prompt}"
            )
        else:
            response_text = (
                f"VLM TEXT RESPONSE: {prompt}"
            )

        return ModelResponse(
            text=response_text,
            model_name=self.model_name,
            model_type=self.model_type,
            sample_id=sample_id,
            image_path=image_path,
        )

    def unload(self) -> None:
        self.loaded = False


class DummyModelRunnerBuilder:
    """
    Builder dummy untuk menguji ExperimentExecutor
    tanpa memuat model nyata.
    """

    def build(self, config):
        if config.model_type == "llm":
            return ModelRunner(DummyLLM())

        return ModelRunner(DummyVLM())


def create_manifest(
    path: Path,
    sample: BenchmarkSample,
    image_path: Path,
) -> None:
    record = {
        "sample_id": sample.sample_id,
        "benchmark": sample.benchmark,
        "image_path": str(image_path),
        "source_prompt": sample.prompt,
        "rendering_config": {
            "width": 1024,
            "height": 1024,
            "font_size": 32,
        },
    }

    import json

    path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )


def create_executor(tmp_path: Path):
    sample = BenchmarkSample(
        sample_id="sample-001",
        benchmark="JBB-Behaviors",
        prompt="Dummy benchmark prompt",
        category="test",
        metadata={
            "test": True,
        },
    )

    image_path = (
        tmp_path / "sample-001.png"
    )

    image_path.write_bytes(
        b"dummy-image-content"
    )

    manifest_path = (
        tmp_path / "visual_manifest.jsonl"
    )

    create_manifest(
        path=manifest_path,
        sample=sample,
        image_path=image_path,
    )

    config = ExperimentConfig(
        model_pair=ModelPairConfig(
            pair_id="dummy-pair",
            llm_model_key="dummy_llm",
            vlm_model_key="dummy_vlm",
        ),
        benchmark="JBB-Behaviors",
        generation=GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            do_sample=False,
        ),
        visual_manifest_path=manifest_path,
        neutral_intro_prompt=(
            "Please describe the content of the image."
        ),
    )

    executor = ExperimentExecutor(
        config=config,
        dataset_samples=[sample],
        model_runner_builder=(
            DummyModelRunnerBuilder()
        ),
        visual_manifest_loader=(
            VisualManifestLoader(manifest_path)
        ),
    )

    return executor, sample, image_path


def test_executor_runs_one_sample(tmp_path):
    executor, sample, image_path = (
        create_executor(tmp_path)
    )

    result = executor.run_sample(
        sample.sample_id
    )

    assert result.sample_id == sample.sample_id
    assert result.complete

    assert result.llm_text is not None
    assert result.vlm_text is not None
    assert result.vlm_image is not None

    assert (
        result.llm_text.response
        == "LLM RESPONSE: Dummy benchmark prompt"
    )

    assert (
        result.vlm_text.response
        == "VLM TEXT RESPONSE: Dummy benchmark prompt"
    )

    assert (
        result.vlm_image.response
        == (
            "VLM IMAGE RESPONSE: "
            "Please describe the content of the image."
        )
    )

    assert (
        result.vlm_image.image_path
        == image_path
    )


def test_executor_preserves_sample_provenance(
    tmp_path,
):
    executor, sample, _ = create_executor(
        tmp_path
    )

    result = executor.run_sample(
        sample.sample_id
    )

    assert (
        result.llm_text.metadata["source_prompt"]
        == sample.prompt
    )

    assert (
        result.vlm_text.metadata["source_prompt"]
        == sample.prompt
    )

    assert (
        result.vlm_image.metadata["source_prompt"]
        == sample.prompt
    )

    assert (
        result.vlm_image.metadata[
            "visual_sample_id"
        ]
        == sample.sample_id
    )


def test_executor_rejects_unknown_sample(
    tmp_path,
):
    executor, _, _ = create_executor(
        tmp_path
    )

    with pytest.raises(ValueError):
        executor.run_sample(
            "sample-does-not-exist"
        )


def test_executor_rejects_missing_visual_sample(
    tmp_path,
):
    sample = BenchmarkSample(
        sample_id="sample-002",
        benchmark="JBB-Behaviors",
        prompt="Another prompt",
        category="test",
    )

    image_path = (
        tmp_path / "sample-001.png"
    )
    image_path.write_bytes(
        b"dummy-image-content"
    )

    manifest_path = (
        tmp_path / "visual_manifest.jsonl"
    )

    create_manifest(
        path=manifest_path,
        sample=BenchmarkSample(
            sample_id="sample-001",
            benchmark="JBB-Behaviors",
            prompt="Different prompt",
        ),
        image_path=image_path,
    )

    config = ExperimentConfig(
        model_pair=ModelPairConfig(
            pair_id="dummy-pair",
            llm_model_key="dummy_llm",
            vlm_model_key="dummy_vlm",
        ),
        benchmark="JBB-Behaviors",
        generation=GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            do_sample=False,
        ),
        visual_manifest_path=manifest_path,
        neutral_intro_prompt=(
            "Please describe the content of the image."
        ),
    )

    executor = ExperimentExecutor(
        config=config,
        dataset_samples=[sample],
        model_runner_builder=(
            DummyModelRunnerBuilder()
        ),
        visual_manifest_loader=(
            VisualManifestLoader(manifest_path)
        ),
    )

    with pytest.raises(ValueError):
        executor.run_sample(
            sample.sample_id
        )


def test_executor_rejects_benchmark_mismatch(
    tmp_path,
):
    sample = BenchmarkSample(
        sample_id="sample-003",
        benchmark="JBB-Behaviors",
        prompt="Benchmark prompt",
        category="test",
    )

    image_path = (
        tmp_path / "sample-003.png"
    )
    image_path.write_bytes(
        b"dummy-image-content"
    )

    manifest_path = (
        tmp_path / "visual_manifest.jsonl"
    )

    visual_sample = VisualSample(
        sample_id=sample.sample_id,
        benchmark="AdvBench",
        image_path=image_path,
        source_prompt=sample.prompt,
    )

    import json

    manifest_path.write_text(
        json.dumps(
            visual_sample.to_dict()
        ) + "\n",
        encoding="utf-8",
    )

    config = ExperimentConfig(
        model_pair=ModelPairConfig(
            pair_id="dummy-pair",
            llm_model_key="dummy_llm",
            vlm_model_key="dummy_vlm",
        ),
        benchmark="JBB-Behaviors",
        generation=GenerationConfig(
            max_new_tokens=32,
            temperature=0.0,
            do_sample=False,
        ),
        visual_manifest_path=manifest_path,
        neutral_intro_prompt=(
            "Please describe the content of the image."
        ),
    )

    executor = ExperimentExecutor(
        config=config,
        dataset_samples=[sample],
        model_runner_builder=(
            DummyModelRunnerBuilder()
        ),
        visual_manifest_loader=(
            VisualManifestLoader(manifest_path)
        ),
    )

    with pytest.raises(ValueError):
        executor.run_sample(
            sample.sample_id
        )