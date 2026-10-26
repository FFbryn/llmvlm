
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
from evaluation.model_runner_builder import ModelRunnerBuilder


# ============================================================
# PROJECT PATHS AND PILOT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_COUNT_PER_DATASET = 3
SELECTION_SEED = 2026
EXPECTED_SELECTION_SIZE = 100

SELECTION_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "selected_semantic_100"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "pilot_batch_3x3"
)

REQUIRED_CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)

# JBB sengaja tidak dimasukkan karena pilot JBB sudah dilakukan.
DATASETS = {
    "advbench": {
        "loader_name": "advbench",
        "selection_jsonl": SELECTION_DIR / "advbench_100.jsonl",
        "visual_manifest": (
            PROJECT_ROOT
            / "benchmark_data"
            / "generated"
            / "visual_manifest_advbench.jsonl"
        ),
        "output": RESULTS_DIR / "advbench_pilot_3.jsonl",
    },
    "harmbench": {
        "loader_name": "harmbench",
        "selection_jsonl": SELECTION_DIR / "harmbench_100.jsonl",
        "visual_manifest": (
            PROJECT_ROOT
            / "benchmark_data"
            / "generated"
            / "visual_manifest_harmbench.jsonl"
        ),
        "output": RESULTS_DIR / "harmbench_pilot_3.jsonl",
    },
    "xstest": {
        "loader_name": "xstest",
        "selection_jsonl": SELECTION_DIR / "xstest_100.jsonl",
        "visual_manifest": (
            PROJECT_ROOT
            / "benchmark_data"
            / "generated"
            / "visual_manifest_xstest.jsonl"
        ),
        "output": RESULTS_DIR / "xstest_pilot_3.jsonl",
    },
}


# ============================================================
# SELECTION FILE VALIDATION
# ============================================================

def load_selected_ids(jsonl_path: Path) -> list[str]:
    """
    Membaca sample_id dari JSONL hasil seleksi 100 sampel.

    Fungsi ini tidak membuat sampel baru dan tidak mengubah
    hasil seleksi. Sampel pilot dipilih dari ID yang tersedia
    di file seleksi tersebut.
    """

    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"File seleksi tidak ditemukan: {jsonl_path}"
        )

    selected_ids = []

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada {jsonl_path}, "
                    f"baris {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record pada {jsonl_path}, baris "
                    f"{line_number} bukan JSON object."
                )

            sample_id = record.get("sample_id")

            if sample_id is None or not str(sample_id).strip():
                raise ValueError(
                    f"sample_id kosong pada {jsonl_path}, "
                    f"baris {line_number}."
                )

            selected_ids.append(str(sample_id).strip())

    if len(selected_ids) != EXPECTED_SELECTION_SIZE:
        raise ValueError(
            f"File seleksi {jsonl_path.name} berisi "
            f"{len(selected_ids)} record, bukan "
            f"{EXPECTED_SELECTION_SIZE} seperti yang diharapkan."
        )

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(
            f"sample_id duplikat ditemukan dalam {jsonl_path}."
        )

    return selected_ids


# ============================================================
# REPRODUCIBLE SAMPLE SELECTION
# ============================================================

def choose_samples(
    dataset_name: str,
    dataset_samples: list,
) -> list:
    """
    Memilih tiga sampel dari 100 sampel hasil seleksi.

    Reproducibility dijaga dengan:
    1. Membaca ID dari file JSONL hasil seleksi.
    2. Mencocokkan ID dengan dataset loader.
    3. Mengurutkan kandidat berdasarkan sample_id.
    4. Menggunakan seed tetap yang berbeda per dataset.
    """

    settings = DATASETS[dataset_name]
    selected_ids = load_selected_ids(
        settings["selection_jsonl"]
    )
    selected_id_set = set(selected_ids)

    samples_by_id = {}

    for sample in dataset_samples:
        sample_id = str(sample.sample_id)

        if sample_id in samples_by_id:
            raise ValueError(
                f"{dataset_name}: sample_id duplikat pada "
                f"dataset loader: {sample_id}"
            )

        samples_by_id[sample_id] = sample

    matched_samples = [
        samples_by_id[sample_id]
        for sample_id in selected_ids
        if sample_id in samples_by_id
    ]

    matched_ids = {
        str(sample.sample_id)
        for sample in matched_samples
    }

    missing_ids = selected_id_set - matched_ids

    if missing_ids:
        preview = sorted(missing_ids)[:10]
        raise RuntimeError(
            f"{dataset_name}: {len(missing_ids)} ID pada file "
            "seleksi tidak ditemukan di dataset loader. "
            f"Contoh ID: {preview}. "
            "Periksa konsistensi sample_id sebelum eksperimen."
        )

    if len(matched_samples) != EXPECTED_SELECTION_SIZE:
        raise RuntimeError(
            f"{dataset_name}: jumlah sampel yang cocok adalah "
            f"{len(matched_samples)}, bukan "
            f"{EXPECTED_SELECTION_SIZE}."
        )

    # Urutan tetap sebelum random sampling.
    matched_samples = sorted(
        matched_samples,
        key=lambda sample: str(sample.sample_id),
    )

    dataset_seed = (
        SELECTION_SEED
        + sum(ord(char) for char in dataset_name)
    )

    rng = random.Random(dataset_seed)

    selected_samples = rng.sample(
        matched_samples,
        SAMPLE_COUNT_PER_DATASET,
    )

    return sorted(
        selected_samples,
        key=lambda sample: str(sample.sample_id),
    )


# ============================================================
# VISUAL MANIFEST VALIDATION
# ============================================================

def verify_manifest(
    dataset_name: str,
    selected_samples: list,
    visual_samples: list,
) -> dict:
    """
    Memastikan setiap sampel terpilih memiliki gambar yang valid
    dan sumber visual cocok dengan sampel benchmark aslinya.
    """

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
                f"{dataset_name}: sample_id {sample_id} "
                "tidak ditemukan di visual manifest."
            )

        if visual_sample.benchmark != sample.benchmark:
            raise RuntimeError(
                f"{sample_id}: benchmark tidak cocok. "
                f"Dataset={sample.benchmark!r}, "
                f"visual={visual_sample.benchmark!r}"
            )

        source_prompt = getattr(
            visual_sample,
            "source_prompt",
            None,
        )

        if (
            source_prompt is not None
            and source_prompt != sample.prompt
        ):
            raise RuntimeError(
                f"{sample_id}: source_prompt pada visual manifest "
                "tidak cocok dengan prompt benchmark."
            )

        image_path = Path(visual_sample.image_path)

        # Path relatif ditafsirkan dari root project.
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        if not image_path.is_file():
            raise FileNotFoundError(
                f"{sample_id}: file gambar tidak ditemukan: "
                f"{image_path}"
            )

    return visual_by_id


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def verify_output(
    output_path: Path,
    expected_ids: set[str],
) -> None:
    """
    Memvalidasi output JSONL setelah batch berjalan.

    Setiap sampel wajib mempunyai tiga kondisi dan respons
    non-empty pada setiap kondisi.
    """

    if not output_path.is_file():
        raise RuntimeError(
            f"File output tidak ditemukan: {output_path}"
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
                    f"JSON tidak valid pada output baris "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise RuntimeError(
                    f"Record output baris {line_number} "
                    "bukan JSON object."
                )

            records.append(record)

    actual_ids = [
        str(record.get("sample_id"))
        for record in records
    ]

    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError(
            f"Output mengandung sample_id duplikat: {output_path}"
        )

    if set(actual_ids) != expected_ids:
        raise RuntimeError(
            f"ID pada output tidak sesuai: {output_path}\n"
            f"Expected: {sorted(expected_ids)}\n"
            f"Actual:   {sorted(actual_ids)}"
        )

    for record in records:
        sample_id = str(record["sample_id"])

        for condition in REQUIRED_CONDITIONS:
            result = record.get(condition)

            if not isinstance(result, dict):
                raise RuntimeError(
                    f"{sample_id}: hasil {condition} "
                    "tidak tersedia atau bukan object."
                )

            if str(result.get("sample_id")) != sample_id:
                raise RuntimeError(
                    f"{sample_id}: sample_id pada kondisi "
                    f"{condition} tidak cocok."
                )

            response = result.get("response")

            if (
                not isinstance(response, str)
                or not response.strip()
            ):
                raise RuntimeError(
                    f"{sample_id}: respons {condition} kosong."
                )

    print(f"  Record terverifikasi: {len(records)}")
    print("  Kondisi per sampel: 3")
    print("  sample_id duplikat: tidak ada")
    print("  Respons kosong: tidak ada")


# ============================================================
# RUN ONE DATASET
# ============================================================

def run_dataset(dataset_name: str) -> dict:
    """Menjalankan pilot batch untuk satu dataset."""

    settings = DATASETS[dataset_name]

    print("\n" + "=" * 72)
    print(f"PILOT BATCH: {dataset_name.upper()}")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. Load original dataset
    # --------------------------------------------------------

    print("\n[1] Memuat dataset asli...")

    dataset_samples = DatasetLoader(
        settings["loader_name"]
    ).load()

    print(f"Jumlah sampel pada loader: {len(dataset_samples)}")

    # --------------------------------------------------------
    # 2. Select 3 from final 100
    # --------------------------------------------------------

    print("\n[2] Memilih 3 sampel dari hasil seleksi 100...")

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
            f"{dataset_name}: terdapat sample_id duplikat "
            "dalam pilihan pilot."
        )

    for sample_id in sample_ids:
        print(f"  - {sample_id}")

    # --------------------------------------------------------
    # 3. Load and validate visual manifest
    # --------------------------------------------------------

    print("\n[3] Memuat dan memeriksa visual manifest...")

    manifest_path = settings["visual_manifest"]

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Visual manifest tidak ditemukan: {manifest_path}\n"
            "Pastikan visual dataset telah dibuat dan "
            "sesuaikan path jika nama file berbeda."
        )

    visual_loader = VisualManifestLoader(manifest_path)
    visual_samples = visual_loader.load()

    print(
        "Jumlah record visual manifest: "
        f"{len(visual_samples)}"
    )

    verify_manifest(
        dataset_name,
        selected_samples,
        visual_samples,
    )

    print("Pemeriksaan manifest dan gambar berhasil.")

    # --------------------------------------------------------
    # 4. Build experiment configuration
    # --------------------------------------------------------

    print("\n[4] Menyiapkan konfigurasi eksperimen...")

    benchmarks = {
        sample.benchmark
        for sample in selected_samples
    }

    if len(benchmarks) != 1:
        raise RuntimeError(
            f"{dataset_name}: sampel terpilih mempunyai "
            f"benchmark berbeda: {sorted(map(str, benchmarks))}"
        )

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

    # --------------------------------------------------------
    # 5. Build executor
    # --------------------------------------------------------

    print("\n[5] Membangun experiment executor...")

    executor = ExperimentExecutor(
        config=config,
        dataset_samples=dataset_samples,
        model_runner_builder=ModelRunnerBuilder(),
        visual_manifest_loader=visual_loader,
    )

    # --------------------------------------------------------
    # 6. Prepare output
    # --------------------------------------------------------

    output_path = settings["output"]
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Resume sengaja dimatikan agar output pilot baru tidak
    # tercampur dengan record dari eksekusi sebelumnya.
    # BatchExperimentRunner akan menolak output yang sudah ada
    # dan tidak kosong, sehingga hasil lama tidak tertimpa.
    if output_path.exists() and output_path.stat().st_size > 0:
        raise FileExistsError(
            f"Output pilot sudah ada dan tidak kosong:\n"
            f"{output_path}\n\n"
            "Periksa hasil sebelumnya. Jika ingin menjalankan "
            "pilot baru, ubah nama output pada DATASETS agar "
            "hasil tidak tercampur atau tertimpa."
        )

    print("\n[6] Menjalankan inferensi model...")
    print(f"Output: {output_path}")
    print(
        "Target kondisi: "
        f"{len(selected_samples)} sampel x "
        f"{len(REQUIRED_CONDITIONS)} kondisi = "
        f"{len(selected_samples) * len(REQUIRED_CONDITIONS)} "
        "hasil kondisi."
    )

    batch_runner = BatchExperimentRunner(
        executor=executor,
        output_path=output_path,
        sample_ids=sample_ids,
        resume=False,
    )

    summary = batch_runner.run()

    print("\nRingkasan batch:")
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    if summary.get("failed") != 0:
        raise RuntimeError(
            f"{dataset_name}: ditemukan "
            f"{summary.get('failed')} sampel gagal. "
            "Periksa ringkasan batch dan error sebelum "
            "melanjutkan ke dataset berikutnya."
        )

    if (
        summary.get("completed", 0)
        + summary.get("skipped", 0)
        != len(sample_ids)
    ):
        raise RuntimeError(
            f"{dataset_name}: jumlah completed + skipped "
            "tidak sama dengan jumlah sampel pilot."
        )

    # --------------------------------------------------------
    # 7. Verify persisted results
    # --------------------------------------------------------

    print("\n[7] Memvalidasi output yang tersimpan...")

    verify_output(
        output_path=output_path,
        expected_ids=set(sample_ids),
    )

    print(f"\nPILOT {dataset_name.upper()} BERHASIL")

    return {
        "dataset": dataset_name,
        "selected_sample_ids": sample_ids,
        "sample_count": len(sample_ids),
        "condition_count": len(REQUIRED_CONDITIONS),
        "expected_condition_results": (
            len(sample_ids) * len(REQUIRED_CONDITIONS)
        ),
        "output_path": str(output_path),
        "status": "passed",
        "batch_summary": summary,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 72)
    print("THREE-DATASET PILOT BATCH")
    print("=" * 72)
    print(f"Sampel per dataset : {SAMPLE_COUNT_PER_DATASET}")
    print(f"Jumlah dataset     : {len(DATASETS)}")
    print(
        "Total sampel target: "
        f"{SAMPLE_COUNT_PER_DATASET * len(DATASETS)}"
    )
    print(f"Seed seleksi       : {SELECTION_SEED}")
    print("Kondisi            : " + ", ".join(REQUIRED_CONDITIONS))
    print("JBB                : tidak dijalankan")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = []

    # Berhenti jika ada dataset gagal agar kegagalan tidak
    # tertutupi oleh hasil dataset berikutnya.
    for dataset_name in DATASETS:
        dataset_summary = run_dataset(dataset_name)
        all_summaries.append(dataset_summary)

    summary_path = RESULTS_DIR / "pilot_summary.json"

    summary_document = {
        "experiment": "pilot_batch_3x3",
        "selection_seed": SELECTION_SEED,
        "samples_per_dataset": SAMPLE_COUNT_PER_DATASET,
        "dataset_count": len(DATASETS),
        "total_samples_target": (
            SAMPLE_COUNT_PER_DATASET * len(DATASETS)
        ),
        "conditions": list(REQUIRED_CONDITIONS),
        "expected_condition_results": (
            SAMPLE_COUNT_PER_DATASET
            * len(DATASETS)
            * len(REQUIRED_CONDITIONS)
        ),
        "datasets": all_summaries,
        "status": "passed",
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_document,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    print("\n" + "=" * 72)
    print("SEMUA PILOT BATCH BERHASIL")
    print("=" * 72)
    print(f"Total sampel target : {summary_document['total_samples_target']}")
    print(
        "Target hasil kondisi: "
        f"{summary_document['expected_condition_results']}"
    )
    print(f"Ringkasan tersimpan : {summary_path}")
    print(f"Folder output       : {RESULTS_DIR}")


if __name__ == "__main__":
    main()
