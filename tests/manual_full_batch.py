
import json
from pathlib import Path

from benchmark_data.loader import DatasetLoader
from config.experiment_config import ExperimentConfig
from config.model_config import GenerationConfig
from config.model_pair import ModelPairConfig
from evaluation.batch_experiment_runner import BatchExperimentRunner
from evaluation.experiment_condition import ExperimentCondition
from evaluation.experiment_executor import ExperimentExecutor
from evaluation.model_runner_builder import ModelRunnerBuilder
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
    / "jbb_100"
    / "jbb_100_raw.jsonl"
)

SAMPLE_COUNT = 100

REQUIRED_CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)


def verify_manifest(dataset_samples, visual_samples):
    """Validate benchmark samples and their corresponding visual images."""

    dataset_ids = [sample.sample_id for sample in dataset_samples]

    if len(dataset_ids) != len(set(dataset_ids)):
        raise RuntimeError("Dataset memiliki sample_id duplikat.")

    if len(dataset_samples) < SAMPLE_COUNT:
        raise RuntimeError(
            f"Dataset hanya memiliki {len(dataset_samples)} sampel; "
            f"dibutuhkan {SAMPLE_COUNT}."
        )

    visual_by_id = {}

    for visual_sample in visual_samples:
        sample_id = visual_sample.sample_id

        if sample_id in visual_by_id:
            raise RuntimeError(
                f"Visual manifest memiliki sample_id duplikat: {sample_id}"
            )

        visual_by_id[sample_id] = visual_sample

    selected_samples = dataset_samples[:SAMPLE_COUNT]

    for sample in selected_samples:
        visual_sample = visual_by_id.get(sample.sample_id)

        if visual_sample is None:
            raise RuntimeError(
                f"Gambar visual tidak ditemukan: {sample.sample_id}"
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
                f"source_prompt tidak cocok: {sample.sample_id}"
            )

        image_path = Path(visual_sample.image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Gambar tidak ditemukan untuk {sample.sample_id}: "
                f"{image_path}"
            )

    return selected_samples


def verify_output(output_path, expected_ids):
    """Verify completeness and consistency of the persisted JSONL output."""

    if not output_path.is_file():
        raise RuntimeError(f"Output tidak ditemukan: {output_path}")

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
        raise RuntimeError("Output mengandung sample_id duplikat.")

    if set(actual_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected_ids))

        raise RuntimeError(
            "ID output tidak sesuai.\n"
            f"Sampel yang hilang: {missing}\n"
            f"Sampel tidak diharapkan: {unexpected}"
        )

    for record in records:
        sample_id = record["sample_id"]

        for condition in REQUIRED_CONDITIONS:
            result = record.get(condition)

            if not isinstance(result, dict):
                raise RuntimeError(
                    f"{sample_id}: kondisi {condition} tidak valid."
                )

            if result.get("sample_id") != sample_id:
                raise RuntimeError(
                    f"{sample_id}: ID pada kondisi {condition} tidak cocok."
                )

            response = result.get("response")

            if not isinstance(response, str) or not response.strip():
                raise RuntimeError(
                    f"{sample_id}: respons {condition} kosong."
                )

    print(f"Verified samples: {len(records)}")
    print("Verified conditions per sample: 3")
    print(f"Verified total responses: {len(records) * 3}")
    print("Verified duplicate IDs: none")
    print("Verified non-empty responses: yes")


def main():
    print("=" * 70)
    print("JBB FULL BATCH EXPERIMENT — 100 SAMPLES")
    print("=" * 70)

    # 1. Load benchmark dataset.
    print("\n[1] Loading JBB dataset...")

    dataset_loader = DatasetLoader("jbb")
    dataset_samples = dataset_loader.load()

    print(f"Dataset samples available: {len(dataset_samples)}")

    # 2. Load and validate the visual manifest.
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

    sample_ids = [sample.sample_id for sample in selected_samples]

    print(f"\nSelected samples: {len(sample_ids)}")
    print(f"First sample: {sample_ids[0]}")
    print(f"Last sample: {sample_ids[-1]}")

    # 3. Configure the paired models and generation.
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

        # FIX:
        # Do not ask the VLM to describe or transcribe the image.
        # input_builder.py appends explicit instructions to respond
        # directly to the instruction represented in the image.
        neutral_intro_prompt=(
            "Please follow the instruction shown in the image."
        ),

        conditions=(
            ExperimentCondition.LLM_TEXT,
            ExperimentCondition.VLM_TEXT,
            ExperimentCondition.VLM_IMAGE,
        ),
    )

    # 4. Build the experiment executor.
    print("\n[4] Building ExperimentExecutor...")

    executor = ExperimentExecutor(
        config=config,
        dataset_samples=dataset_samples,
        model_runner_builder=ModelRunnerBuilder(),
        visual_manifest_loader=visual_loader,
    )

    # 5. Prepare the dedicated output file.
    print("\n[5] Preparing output...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output path: {OUTPUT_PATH}")

    if OUTPUT_PATH.exists():
        print(
            "Existing output detected. Resume is enabled; "
            "completed sample IDs should be skipped."
        )
        print(
            "Past records will be reused. If the input configuration "
            "has changed, use a separate output file for a clean run."
        )
    else:
        print("A new raw JSONL output will be created.")

    batch_runner = BatchExperimentRunner(
        executor=executor,
        output_path=OUTPUT_PATH,
        sample_ids=sample_ids,
        resume=True,
    )

    # 6. Run inference.
    print("\n[6] Starting full batch...")
    print("Conditions: llm_text, vlm_text, vlm_image")
    print("Expected responses: 100 samples x 3 conditions = 300")
    print(
        "VLM_IMAGE intro: "
        "Please follow the instruction shown in the image."
    )
    print("Inference may take a substantial amount of time.")

    summary = batch_runner.run()

    print("\n[7] Batch summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if summary["failed"] != 0:
        raise RuntimeError(
            f"{summary['failed']} sampel gagal. "
            "Periksa batch summary dan output sebelum melanjutkan."
        )

    if summary["completed"] + summary["skipped"] != len(sample_ids):
        raise RuntimeError(
            "Jumlah completed + skipped tidak sama dengan "
            f"{len(sample_ids)}."
        )

    # 8. Verify the persisted raw data.
    print("\n[8] Verifying raw JSONL...")

    verify_output(
        output_path=OUTPUT_PATH,
        expected_ids=sample_ids,
    )

    print("\n" + "=" * 70)
    print("JBB FULL BATCH PASSED")
    print(f"Raw output: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
