
import csv
import json
import random
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

SAMPLE_COUNT_PER_DATASET = 5
SELECTION_SEED = 2026

REQUIRED_CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "smoke_test_15"
)

# Sesuaikan nama manifest jika nama file di project berbeda.
DATASETS = {
    "advbench": {
        "loader_name": "advbench",
        "visual_manifest": (
            PROJECT_ROOT
            / "benchmark_data"
            / "generated"
            / "visual_manifest_advbench.jsonl"
        ),
        "selected_ids_csv": None,
        "output": RESULTS_DIR / "advbench_smoke_5.jsonl",
    },
    "harmbench": {
        "loader_name": "harmbench",
        "visual_manifest": (
            PROJECT_ROOT
            / "benchmark_data"
            / "generated"
            / "visual_manifest_harmbench.jsonl"
        ),
        "selected_ids_csv": (
            PROJECT_ROOT
            / "benchmark_data"
            / "results"
            / "dataset_overlap"
            / "harmbench_selected_unique.csv"
        ),
        "output": RESULTS_DIR / "harmbench_smoke_5.jsonl",
    },
    "xstest": {
        "loader_name": "xstest",
        "visual_manifest": (
            PROJECT_ROOT
            / "benchmark_data"
            / "generated"
            / "visual_manifest_xstest.jsonl"
        ),
        "selected_ids_csv": (
            PROJECT_ROOT
            / "benchmark_data"
            / "results"
            / "dataset_overlap"
            / "xstest_selected_unique.csv"
        ),
        "output": RESULTS_DIR / "xstest_smoke_5.jsonl",
    },
}


def load_selected_ids(csv_path: Path) -> set[str]:
    """Baca ID sampel yang lolos seleksi duplikasi."""

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"File seleksi tidak ditemukan: {csv_path}"
        )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise ValueError(
                f"Kolom sample_id tidak ditemukan di {csv_path}. "
                f"Kolom tersedia: {reader.fieldnames}"
            )

        selected_ids = {
            row["sample_id"].strip()
            for row in reader
            if row.get("sample_id")
            and row["sample_id"].strip()
        }

    if not selected_ids:
        raise RuntimeError(
            f"Tidak ada sample_id yang valid di {csv_path}"
        )

    return selected_ids


def choose_samples(dataset_name, dataset_samples):
    """
    Pilih lima sampel secara reproducible.

    HarmBench dan XSTest dibatasi pada ID yang lolos
    pemeriksaan duplikasi sebelumnya.
    """

    if len(dataset_samples) == 0:
        raise RuntimeError(
            f"Dataset {dataset_name} tidak memiliki sampel."
        )

    selected_csv = DATASETS[dataset_name]["selected_ids_csv"]

    if selected_csv is None:
        candidates = list(dataset_samples)
    else:
        selected_ids = load_selected_ids(selected_csv)

        candidates = [
            sample
            for sample in dataset_samples
            if str(sample.sample_id) in selected_ids
        ]

        if not candidates:
            raise RuntimeError(
                f"Tidak ada sampel {dataset_name} yang cocok "
                f"dengan ID dalam {selected_csv}."
            )

        matched_ids = {
            str(sample.sample_id)
            for sample in candidates
        }

        unmatched_ids = selected_ids - matched_ids

        if unmatched_ids:
            print(
                f"WARNING: {len(unmatched_ids)} ID pada file seleksi "
                f"tidak ditemukan dalam adapter {dataset_name}."
            )

    # Urutkan dahulu agar pemilihan tidak bergantung pada urutan loader.
    candidates = sorted(
        candidates,
        key=lambda sample: str(sample.sample_id),
    )

    if len(candidates) < SAMPLE_COUNT_PER_DATASET:
        raise RuntimeError(
            f"{dataset_name} hanya memiliki {len(candidates)} "
            f"kandidat; dibutuhkan {SAMPLE_COUNT_PER_DATASET}."
        )

    # Seed berbeda per dataset, tetapi tetap dapat direproduksi.
    dataset_seed = (
        SELECTION_SEED
        + sum(ord(char) for char in dataset_name)
    )

    rng = random.Random(dataset_seed)

    return rng.sample(
        candidates,
        SAMPLE_COUNT_PER_DATASET,
    )


def verify_manifest(dataset_name, selected_samples, visual_samples):
    """Periksa keberadaan gambar dan kecocokan sumbernya."""

    visual_by_id = {}

    for visual_sample in visual_samples:
        sample_id = str(visual_sample.sample_id)

        if sample_id in visual_by_id:
            raise RuntimeError(
                f"{dataset_name}: sample_id duplikat dalam "
                f"visual manifest: {sample_id}"
            )

        visual_by_id[sample_id] = visual_sample

    for sample in selected_samples:
        sample_id = str(sample.sample_id)
        visual_sample = visual_by_id.get(sample_id)

        if visual_sample is None:
            raise RuntimeError(
                f"{dataset_name}: gambar tidak terdaftar untuk "
                f"sample_id {sample_id}."
            )

        if visual_sample.benchmark != sample.benchmark:
            raise RuntimeError(
                f"{sample_id}: benchmark tidak cocok. "
                f"Dataset={sample.benchmark!r}, "
                f"visual={visual_sample.benchmark!r}"
            )

        if (
            visual_sample.source_prompt is not None
            and visual_sample.source_prompt != sample.prompt
        ):
            raise RuntimeError(
                f"{sample_id}: source_prompt tidak cocok "
                "dengan prompt asli."
            )

        image_path = Path(visual_sample.image_path)

        # Path relatif ditafsirkan dari root project.
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        if not image_path.is_file():
            raise FileNotFoundError(
                f"{sample_id}: gambar tidak ditemukan: {image_path}"
            )

    return visual_by_id


def verify_output(output_path: Path, expected_ids: set[str]):
    """Validasi JSONL dan tiga respons untuk setiap sampel."""

    if not output_path.is_file():
        raise RuntimeError(
            f"Output tidak ditemukan: {output_path}"
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
                    f"Record pada baris {line_number} "
                    "bukan JSON object."
                )

            records.append(record)

    actual_ids = [
        str(record.get("sample_id"))
        for record in records
    ]

    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError(
            f"Output memiliki sample_id duplikat: {output_path}"
        )

    if set(actual_ids) != expected_ids:
        raise RuntimeError(
            f"ID output tidak sesuai: {output_path}\n"
            f"Expected: {sorted(expected_ids)}\n"
            f"Actual:   {sorted(actual_ids)}"
        )

    for record in records:
        sample_id = str(record["sample_id"])

        for condition in REQUIRED_CONDITIONS:
            result = record.get(condition)

            if not isinstance(result, dict):
                raise RuntimeError(
                    f"{sample_id}: hasil {condition} tidak tersedia."
                )

            if str(result.get("sample_id")) != sample_id:
                raise RuntimeError(
                    f"{sample_id}: ID pada hasil {condition} "
                    "tidak cocok."
                )

            response = result.get("response")

            if not isinstance(response, str) or not response.strip():
                raise RuntimeError(
                    f"{sample_id}: respons {condition} kosong."
                )

    print(f"  Verified records: {len(records)}")
    print("  Verified conditions: 3 per sample")
    print("  Verified duplicate IDs: none")
    print("  Verified non-empty responses: yes")


def run_dataset(dataset_name: str):
    """Jalankan smoke test untuk satu dataset."""

    settings = DATASETS[dataset_name]

    print("\n" + "=" * 70)
    print(f"SMOKE TEST: {dataset_name.upper()}")
    print("=" * 70)

    # 1. Muat dataset asli agar metadata tetap terjaga.
    print("\n[1] Loading original dataset...")

    dataset_samples = DatasetLoader(
        settings["loader_name"]
    ).load()

    print(f"Available samples: {len(dataset_samples)}")

    # 2. Pilih lima sampel.
    print("\n[2] Selecting samples...")

    selected_samples = choose_samples(
        dataset_name,
        dataset_samples,
    )

    sample_ids = [
        str(sample.sample_id)
        for sample in selected_samples
    ]

    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError(
            f"{dataset_name}: ID duplikat dalam pilihan sampel."
        )

    print(f"Selected samples: {len(selected_samples)}")

    for sample_id in sample_ids:
        print(f"  - {sample_id}")

    # 3. Muat dan validasi visual manifest.
    print("\n[3] Loading visual manifest...")

    manifest_path = settings["visual_manifest"]

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Visual manifest belum tersedia: {manifest_path}\n"
            "Buat visual dataset dan manifest terlebih dahulu, "
            "atau sesuaikan path pada DATASETS."
        )

    visual_loader = VisualManifestLoader(manifest_path)
    visual_samples = visual_loader.load()

    print(f"Visual samples available: {len(visual_samples)}")

    verify_manifest(
        dataset_name,
        selected_samples,
        visual_samples,
    )

    print("Visual manifest verification passed.")

    # 4. Konfigurasi model dan generasi.
    print("\n[4] Building experiment configuration...")

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
        benchmark=selected_samples[0].benchmark,
        generation=generation,
        visual_manifest_path=manifest_path,
        neutral_intro_prompt=(
            "Please describe the content of the image."
        ),
        conditions=(
            ExperimentCondition.LLM_TEXT,
            ExperimentCondition.VLM_TEXT,
            ExperimentCondition.VLM_IMAGE,
        ),
    )

    # 5. Bangun executor menggunakan dataset asli.
    print("\n[5] Building executor...")

    from evaluation.model_runner_builder import ModelRunnerBuilder

    executor = ExperimentExecutor(
        config=config,
        dataset_samples=dataset_samples,
        model_runner_builder=ModelRunnerBuilder(),
        visual_manifest_loader=visual_loader,
    )

    # 6. Jalankan batch.
    output_path = settings["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n[6] Starting real model inference...")
    print(f"Output: {output_path}")
    print("Target: 5 samples x 3 conditions")

    batch_runner = BatchExperimentRunner(
        executor=executor,
        output_path=output_path,
        sample_ids=sample_ids,
        resume=True,
    )

    summary = batch_runner.run()

    print("\n[7] Batch summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if summary["failed"] != 0:
        raise RuntimeError(
            f"{dataset_name}: {summary['failed']} sampel gagal. "
            "Periksa batch summary dan error sebelum melanjutkan."
        )

    if summary["completed"] + summary["skipped"] != len(sample_ids):
        raise RuntimeError(
            f"{dataset_name}: jumlah completed + skipped "
            "tidak sesuai jumlah sampel."
        )

    # 7. Verifikasi hasil yang tersimpan.
    print("\n[8] Verifying persisted results...")

    verify_output(
        output_path=output_path,
        expected_ids=set(sample_ids),
    )

    print(f"\n{dataset_name.upper()} SMOKE TEST PASSED")


def main():
    print("=" * 70)
    print("THREE-DATASET REAL BATCH SMOKE TEST")
    print(f"Samples per dataset: {SAMPLE_COUNT_PER_DATASET}")
    print(f"Total target samples: {SAMPLE_COUNT_PER_DATASET * len(DATASETS)}")
    print(f"Selection seed: {SELECTION_SEED}")
    print("=" * 70)

    for dataset_name in DATASETS:
        run_dataset(dataset_name)

    print("\n" + "=" * 70)
    print("ALL THREE DATASET SMOKE TESTS PASSED")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
