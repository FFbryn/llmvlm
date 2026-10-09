from pathlib import Path

from benchmark_data.loader import DatasetLoader
from config.experiment_config import ExperimentConfig
from config.model_config import GenerationConfig
from config.model_pair import ModelPairConfig
from evaluation.batch_experiment_runner import BatchExperimentRunner
from evaluation.experiment_executor import ExperimentExecutor
from preprocessing.visual_manifest_loader import VisualManifestLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VISUAL_MANIFEST = (
    PROJECT_ROOT
    / "benchmark_data"
    / "generated"
    / "visual_manifest_jbb.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "smoke_test"
    / "jbb_real_batch.jsonl"
)


def main():
    print("=" * 70)
    print("REAL BATCH SMOKE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------
    print("\n[1] Loading JBB dataset...")

    dataset_loader = DatasetLoader("jbb")
    dataset_samples = dataset_loader.load()

    print(f"Dataset samples available: {len(dataset_samples)}")

    if len(dataset_samples) < 2:
        raise RuntimeError(
            "Dataset JBB memiliki kurang dari 2 sample."
        )

    # ---------------------------------------------------------
    # 2. Check visual manifest
    # ---------------------------------------------------------
    print("\n[2] Loading visual manifest...")

    if not VISUAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"Visual manifest tidak ditemukan:\n{VISUAL_MANIFEST}"
        )

    visual_loader = VisualManifestLoader(VISUAL_MANIFEST)
    visual_samples = visual_loader.load()

    print(f"Visual samples available: {len(visual_samples)}")

    if len(visual_samples) == 0:
        raise RuntimeError(
            "Visual manifest tidak memiliki sample."
        )

    # ---------------------------------------------------------
    # 3. Experiment configuration
    # ---------------------------------------------------------
    print("\n[3] Building experiment configuration...")

    model_pair = ModelPairConfig(
        pair_id="qwen_qwen",
        llm_model_key="qwen_llm",
        vlm_model_key="qwen_vlm",
    )

    generation = GenerationConfig(
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
    )

    config = ExperimentConfig(
        model_pair=model_pair,
        benchmark="JBB-Behaviors",
        generation=generation,
        visual_manifest_path=VISUAL_MANIFEST,
        neutral_intro_prompt=(
            "Please describe the content of the image."
        ),
        conditions=(
            ExperimentCondition.LLM_TEXT,
            ExperimentCondition.VLM_TEXT,
            ExperimentCondition.VLM_IMAGE,
        ),
    )

    # ---------------------------------------------------------
    # 4. Build executor
    # ---------------------------------------------------------
    print("\n[4] Building ExperimentExecutor...")

    from config.model_config_builder import ModelConfigBuilder
    from models.runner_builder import ModelRunnerBuilder

    model_config_builder = ModelConfigBuilder(
        model_pair=config.model_pair,
        generation=config.generation,
        device=config.device,
        torch_dtype=config.torch_dtype,
    )

    model_runner_builder = ModelRunnerBuilder()

    executor = ExperimentExecutor(
        config=config,
        dataset_samples=dataset_samples,
        model_runner_builder=model_runner_builder,
        visual_manifest_loader=visual_loader,
    )

    # ---------------------------------------------------------
    # 5. Create batch runner
    # ---------------------------------------------------------
    print("\n[5] Creating BatchExperimentRunner...")

    batch_runner = BatchExperimentRunner(
        executor=executor,
        output_path=OUTPUT_PATH,
        sample_ids=[
            dataset_samples[0].sample_id,
            dataset_samples[1].sample_id,
        ],
        resume=True,
    )

    # ---------------------------------------------------------
    # 6. Run
    # ---------------------------------------------------------
    print("\n[6] Starting real batch experiment...")
    print(f"Output: {OUTPUT_PATH}")
    print()

    batch_runner.run()

    # ---------------------------------------------------------
    # 7. Verify output
    # ---------------------------------------------------------
    print("\n[7] Verifying output...")

    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            "Batch selesai tetapi output JSONL tidak ditemukan."
        )

    lines = [
        line
        for line in OUTPUT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print(f"Output records: {len(lines)}")

    if len(lines) == 0:
        raise RuntimeError(
            "Output JSONL kosong."
        )

    print("\n" + "=" * 70)
    print("REAL BATCH SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()