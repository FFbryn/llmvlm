
import json
from pathlib import Path

from benchmark_data.loader import DatasetLoader

from preprocessing.dataset_to_image import (
    DatasetImagePreprocessor,
)

from preprocessing.text_to_image import (
    TextToImageRenderer,
)

from preprocessing.visual_manifest import (
    VisualManifestWriter,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "selected_semantic_100"
)

GENERATED_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "generated"
)

IMAGE_OUTPUT_DIR = (
    GENERATED_DIR
    / "text_rendered"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "advbench": RESULTS_DIR / "advbench_100.jsonl",
    "harmbench": RESULTS_DIR / "harmbench_100.jsonl",
    "xstest": RESULTS_DIR / "xstest_100.jsonl",
}


# ============================================================
# RENDERING CONFIGURATION
# ============================================================

RENDERING_CONFIG = {
    "width": 1024,
    "height": 1024,
    "background": "white",
    "text_color": "black",
    "margin": 64,
    "font_size": 32,
    "line_spacing": 10,
}


# ============================================================
# LOAD SELECTED SAMPLE IDS
# ============================================================

def load_selected_ids(
    jsonl_path: Path,
) -> list[str]:

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"File seleksi tidak ditemukan:\n"
            f"{jsonl_path}"
        )

    selected_ids = []

    with jsonl_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada "
                    f"baris {line_number}: {jsonl_path}"
                ) from exc

            sample_id = record.get("sample_id")

            if not sample_id:
                raise ValueError(
                    f"sample_id tidak ditemukan "
                    f"pada baris {line_number}: "
                    f"{jsonl_path}"
                )

            selected_ids.append(str(sample_id))

    if len(selected_ids) != 100:
        raise ValueError(
            f"Jumlah sampel pada {jsonl_path.name} "
            f"bukan 100.\n"
            f"Jumlah ditemukan: {len(selected_ids)}"
        )

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(
            f"Ditemukan sample_id duplikat "
            f"pada {jsonl_path.name}"
        )

    return selected_ids


# ============================================================
# BUILD VISUAL DATASET
# ============================================================

def build_selected_visual_dataset(
    benchmark: str,
    selection_path: Path,
) -> dict:

    print("\n" + "-" * 70)
    print(f"MEMPROSES {benchmark.upper()}")
    print("-" * 70)

    # --------------------------------------------------------
    # 1. Load selected 100 IDs
    # --------------------------------------------------------

    print("\n[1] Membaca hasil seleksi 100...")

    selected_ids = load_selected_ids(
        selection_path
    )

    selected_id_set = set(selected_ids)

    print(
        f"Jumlah sampel terpilih: {len(selected_ids)}"
    )

    # --------------------------------------------------------
    # 2. Load original benchmark
    # --------------------------------------------------------

    print("\n[2] Memuat dataset asli...")

    loader = DatasetLoader(benchmark)

    original_samples = loader.load()

    sample_index = {}

    for sample in original_samples:

        if sample.sample_id in sample_index:
            raise ValueError(
                f"ID duplikat pada dataset asli "
                f"{benchmark}: {sample.sample_id}"
            )

        sample_index[sample.sample_id] = sample

    # --------------------------------------------------------
    # 3. Match selected IDs with original samples
    # --------------------------------------------------------

    print("\n[3] Mencocokkan sampel...")

    missing_ids = (
        selected_id_set
        - set(sample_index.keys())
    )

    if missing_ids:
        raise ValueError(
            f"Ada {len(missing_ids)} ID seleksi "
            f"yang tidak ditemukan dalam dataset asli "
            f"{benchmark}:\n"
            f"{sorted(missing_ids)}"
        )

    selected_samples = [
        sample_index[sample_id]
        for sample_id in selected_ids
    ]

    print(
        f"Sampel berhasil dicocokkan: "
        f"{len(selected_samples)}"
    )

    # --------------------------------------------------------
    # 4. Initialize renderer
    # --------------------------------------------------------

    print("\n[4] Menyiapkan image renderer...")

    renderer = TextToImageRenderer(
        **RENDERING_CONFIG
    )

    preprocessor = DatasetImagePreprocessor(
        renderer=renderer,
        output_dir=IMAGE_OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # 5. Render selected samples
    # --------------------------------------------------------

    print("\n[5] Membuat gambar...")

    visual_samples = []

    for index, sample in enumerate(
        selected_samples,
        start=1,
    ):

        visual_sample = preprocessor.process_sample(
            sample
        )

        visual_samples.append(
            visual_sample
        )

        print(
            f"[{index}/{len(selected_samples)}] "
            f"{sample.sample_id}"
        )

    # --------------------------------------------------------
    # 6. Generate visual manifest
    # --------------------------------------------------------

    print("\n[6] Membuat visual manifest...")

    manifest_path = (
        GENERATED_DIR
        / f"visual_manifest_{benchmark}.jsonl"
    )

    writer = VisualManifestWriter(
        output_path=manifest_path
    )

    writer.write(visual_samples)

    # --------------------------------------------------------
    # 7. Validate generated images and manifest
    # --------------------------------------------------------

    print("\n[7] Memvalidasi hasil...")

    for visual_sample in visual_samples:

        if not visual_sample.image_path.exists():
            raise FileNotFoundError(
                f"Gambar tidak ditemukan: "
                f"{visual_sample.image_path}"
            )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        manifest_records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    manifest_ids = [
        record["sample_id"]
        for record in manifest_records
    ]

    if len(manifest_records) != 100:
        raise RuntimeError(
            f"Manifest {benchmark} tidak berisi "
            f"tepat 100 record."
        )

    if manifest_ids != selected_ids:
        raise RuntimeError(
            f"Urutan atau ID manifest tidak sesuai "
            f"dengan hasil seleksi {benchmark}."
        )

    print(f"\nDataset             : {benchmark}")
    print(f"Selected samples    : {len(selected_samples)}")
    print(f"Generated images    : {len(visual_samples)}")
    print(f"Manifest records    : {len(manifest_records)}")
    print(f"Manifest path       : {manifest_path}")
    print("Validation          : PASSED")

    return {
        "benchmark": benchmark,
        "selected": len(selected_samples),
        "images": len(visual_samples),
        "manifest": len(manifest_records),
        "manifest_path": manifest_path,
        "status": "SUCCESS",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SELECTED 100 VISUAL DATASET GENERATION")
    print("=" * 70)

    print("\nSumber:")
    print(RESULTS_DIR)

    print("\nOutput gambar:")
    print(IMAGE_OUTPUT_DIR)

    GENERATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMAGE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for index, (benchmark, selection_path) in enumerate(
        DATASETS.items(),
        start=1,
    ):

        print("\n" + "=" * 70)

        print(
            f"[{index}/{len(DATASETS)}] "
            f"{benchmark.upper()}"
        )

        print("=" * 70)

        result = build_selected_visual_dataset(
            benchmark=benchmark,
            selection_path=selection_path,
        )

        results.append(result)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL VISUAL DATASET SUMMARY")
    print("=" * 70)

    total_selected = 0
    total_images = 0
    total_manifests = 0

    for result in results:

        print(f"\nDataset: {result['benchmark']}")

        print(f"Selected : {result['selected']}")
        print(f"Images   : {result['images']}")
        print(f"Manifest : {result['manifest']}")
        print(f"Status   : {result['status']}")

        total_selected += result["selected"]
        total_images += result["images"]
        total_manifests += result["manifest"]

    print("\n" + "-" * 70)

    print(f"Total selected samples : {total_selected}")
    print(f"Total generated images : {total_images}")
    print(f"Total manifest records : {total_manifests}")

    print("\nSTATUS: SUCCESS")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
