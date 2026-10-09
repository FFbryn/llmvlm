
import json
from pathlib import Path

from benchmark_data.loader import DatasetLoader
from config.experiment_config import ExperimentConfig
from config.model_config import GenerationConfig
from config.model_pair import ModelPairConfig
from evaluation.batch_experiment_runner import BatchExperimentRunner
from evaluation.experiment_condition import ExperimentCondition
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
    / "jbb_real_batch_smoke_5.jsonl"
)

SMOKE_SAMPLE_COUNT = 5

REQUIRED_CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)


def verify_manifest(dataset_samples, visual_samples):
    """Validasi ID visual dan kesesuaiannya dengan dataset."""

    dataset_by_id = {
        sample.sample_id: sample
        for sample in dataset_samples
    }

    visual_by_id = {}

    for visual_sample in visual_samples:
        sample_id = visual_sample.sample_id

        if sample_id in visual_by_id:
            raise RuntimeError(
                f"sample_id duplikat pada visual manifest: {sample_id}"
            )

        visual_by_id[sample_id] = visual_sample

    selected_samples = dataset_samples[:SMOKE_SAMPLE_COUNT]

    if len(selected_samples) < SMOKE_SAMPLE_COUNT:
        raise RuntimeError(
            f"Dataset hanya memiliki {len(selected_samples)} sample; "
            f"dibutuhkan {SMOKE_SAMPLE_COUNT}."
        )

    for sample in selected_samples:
        visual_sample = visual_by_id.get(sample.sample_id)

        if visual_sample is None:
            raise RuntimeError(
                f"Visual sample tidak ditemukan: {sample.sample_id}"
            )

        if visual_sample.benchmark != sample.benchmark:
            raise RuntimeError(
                f"Benchmark tidak cocok untuk {sample.sample_id}: "
                f"dataset={sample.benchmark!r}, "
                f"visual={visual_sample.benchmark!r}"
            )

        if (
            visual_sample.source_prompt is not None
            and visual_sample.source_prompt != sample.prompt
        ):
            raise RuntimeError(
                f"source_prompt tidak cocok untuk {sample.sample_id}"
            )

        image_path = Path(visual_sample.image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Gambar tidak ditemukan untuk {sample.sample_id}: "
                f"{image_path}"
            )

    return selected_samples


def verify_output(output_path, expected_ids):
    """Validasi struktur JSONL setelah batch selesai."""

    if not output_path.is_file():
        raise RuntimeError(
            f"Output JSONL tidak ditemukan: {output_path}"
        )

    records = []

    with output_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"JSON tidak valid pada baris {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Record baris {line_number} bukan object JSON."
                )

            records.append(record)

    actual_ids = [record.get("sample_id") for record in records]

    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError(
            "Output mengandung sample_id duplikat."
        )

    if set(actual_ids) != set(expected_ids):
        raise RuntimeError(
            "ID output tidak cocok dengan ID yang diuji.\n"
            f"Expected: {sorted(expected_ids)}\n"
            f"Actual:   {sorted(actual_ids)}"
        )

    for record in records:
        sample_id = record["sample_id"]

        for condition in REQUIRED_CONDITIONS:
            result = record.get(condition)

            if not isinstance(result, dict):
                raise RuntimeError(
                    f"{sample_id}: kondisi {condition} tidak tersedia "
                    "atau bukan object JSON."
                )

            if result.get("sample_id") != sample_id:
                raise RuntimeError(
                    f"{sample_id}: sample_id pada {condition} tidak cocok."
                )

            response = result.get("response")

            if not isinstance(response, str) or not response.strip():
                raise RuntimeError(
                    f"{sample_id}: respons {condition} kosong atau tidak valid."
                )

    print(f"Verified records: {len(records)}")
    print("Verified conditions per record: 3")
    print("Verified duplicate sample IDs: none")
    print("Verified non-empty responses: yes")


def main():
    print("=" * 70)
    print("REAL BATCH SMOKE TEST — 5 SAMPLES")
    print("=" * 70)

    # 1. Load dataset.
    print("\n[1] Loading JBB dataset...")

    dataset_loader = DatasetLoader("jbb")
    dataset_samples = dataset_loader.load()

    print(f"Dataset samples available: {len(dataset_samples)}")

    if len(dataset_samples) < SMOKE_SAMPLE_COUNT:
        raise RuntimeError(
            f"Dibutuhkan minimal {SMOKE_SAMPLE_COUNT} sample JBB."
        )

    # 2. Load visual manifest.
    print("\n[2] Loading visual manifest...")

    if not VISUAL_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Visual manifest tidak ditemukan: {VISUAL_MANIFEST}"
        )

    visual_loader = VisualManifestLoader(VISUAL_MANIFEST)
    visual_samples = visual_loader.load()

    print(f"Visual samples available: {len(visual_samples)}")

    selected_samples = verify_manifest(
        dataset_samples,
        visual_samples,
    )

    sample_ids = [
        sample.sample_id
        for sample in selected_samples
    ]

    print(f"Selected samples: {len(sample_ids)}")

    for sample_id in sample_ids:
        print(f"  - {sample_id}")

    # 3. Build experiment configuration.
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
        neutral_intro_prompt="Please describe the content of the image.",
        conditions=(
            ExperimentCondition.LLM_TEXT,
            ExperimentCondition.VLM_TEXT,
            ExperimentCondition.VLM_IMAGE,
        ),
    )

    # 4. Build executor.
    print("\n[4] Building ExperimentExecutor...")

    from evaluation.model_runner_builder import ModelRunnerBuilder

    executor = ExperimentExecutor(
        config=config,
        dataset_samples=dataset_samples,
        model_runner_builder=ModelRunnerBuilder(),
        visual_manifest_loader=visual_loader,
    )

    # 5. Build batch runner.
    print("\n[5] Creating BatchExperimentRunner...")

    print(f"Output path: {OUTPUT_PATH}")

    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size == 0:
        print("Output file exists but is empty.")

    batch_runner = BatchExperimentRunner(
        executor=executor,
        output_path=OUTPUT_PATH,
        sample_ids=sample_ids,
        resume=True,
    )

    # 6. Run actual model inference.
    print("\n[6] Starting real batch experiment...")
    print("The LLM and VLM will be loaded for the batch session.")
    print("This may require substantial RAM or GPU memory.")

    summary = batch_runner.run()

    print("\n[7] Checking batch summary...")

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if summary["failed"] != 0:
        raise RuntimeError(
            f"{summary['failed']} sample gagal. "
            "Periksa error pada batch summary sebelum melanjutkan."
        )

    if summary["completed"] + summary["skipped"] != len(sample_ids):
        raise RuntimeError(
            "Jumlah sample completed + skipped tidak sesuai "
            "dengan jumlah sample yang dipilih."
        )

    # 8. Verify persisted results.
    print("\n[8] Verifying JSONL output...")

    verify_output(
        output_path=OUTPUT_PATH,
        expected_ids=sample_ids,
    )

    print("\n" + "=" * 70)
    print("REAL BATCH SMOKE TEST PASSED")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
