
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from benchmark_data.loader import DatasetLoader


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "dataset_overlap"
)

EXACT_CSV = OUTPUT_DIR / "harmbench_existing_exact_duplicates.csv"
NORMALIZED_CSV = OUTPUT_DIR / "harmbench_existing_normalized_duplicates.csv"
SELECTED_CSV = OUTPUT_DIR / "harmbench_selected_unique.csv"
REPORT_JSON = OUTPUT_DIR / "harmbench_overlap_report.json"


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_DATASETS = ("advbench", "jbb")
TARGET_DATASET = "harmbench"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_prompt(prompt: str) -> str:
    """
    Menormalkan kapitalisasi dan whitespace.

    Tanda baca, urutan kata, dan isi prompt tetap dipertahankan.
    Ini bukan pemeriksaan kemiripan semantik.
    """
    return " ".join(prompt.casefold().split())


# ============================================================
# DATASET LOADING AND VALIDATION
# ============================================================

def load_samples(dataset_name: str):
    """
    Memuat dataset menggunakan DatasetLoader project.
    """
    loader = DatasetLoader(dataset_name)
    return loader.load()


def extract_sample_fields(sample) -> Tuple[str, str]:
    """
    Mengambil sample_id dan prompt dari BenchmarkSample.
    """
    sample_id = getattr(sample, "sample_id", None)
    prompt = getattr(sample, "prompt", None)

    if sample_id is None:
        raise ValueError(
            "Sample tidak memiliki atribut 'sample_id'. "
            "Periksa benchmark_data/schema.py."
        )

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"Prompt kosong atau tidak valid pada sampel {sample_id!r}."
        )

    return str(sample_id), prompt


def validate_samples(samples, dataset_name: str) -> List[Dict[str, str]]:
    """
    Memastikan ID unik dalam dataset dan prompt valid.
    """
    records = []
    seen_ids = set()

    for sample in samples:
        sample_id, prompt = extract_sample_fields(sample)

        if sample_id in seen_ids:
            raise ValueError(
                f"ID duplikat pada dataset {dataset_name}: {sample_id}"
            )

        seen_ids.add(sample_id)

        records.append({
            "dataset": dataset_name,
            "sample_id": sample_id,
            "prompt": prompt,
        })

    return records


# ============================================================
# OVERLAP DETECTION
# ============================================================

def find_matches(
    target_records: List[Dict[str, str]],
    reference_records: List[Dict[str, str]],
    normalized: bool,
) -> List[Dict[str, str]]:
    """
    Membandingkan dataset target dengan satu dataset referensi.

    Satu sampel dapat memiliki lebih dari satu pasangan cocok.
    Semua pasangan disimpan agar informasi tidak hilang.
    """
    key_function = normalize_prompt if normalized else lambda text: text

    reference_index = defaultdict(list)

    for record in reference_records:
        key = key_function(record["prompt"])

        if key:
            reference_index[key].append(record)

    matches = []

    for target in target_records:
        key = key_function(target["prompt"])

        if not key:
            continue

        for reference in reference_index.get(key, []):
            matches.append({
                "target_dataset": TARGET_DATASET,
                "target_sample_id": target["sample_id"],
                "reference_dataset": reference["dataset"],
                "reference_sample_id": reference["sample_id"],
                "target_prompt": target["prompt"],
                "reference_prompt": reference["prompt"],
            })

    return matches


def write_matches_csv(
    output_path: Path,
    matches: List[Dict[str, str]],
) -> None:
    """
    Menyimpan seluruh pasangan cocok dalam CSV.
    """
    fieldnames = [
        "target_dataset",
        "target_sample_id",
        "reference_dataset",
        "reference_sample_id",
        "target_prompt",
        "reference_prompt",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)


def matched_target_ids(
    matches: List[Dict[str, str]],
) -> set:
    """
    Mengambil ID HarmBench unik yang memiliki pasangan cocok.
    """
    return {
        row["target_sample_id"]
        for row in matches
    }


# ============================================================
# SELECTION
# ============================================================

def select_unique_samples(
    harmbench: List[Dict[str, str]],
    normalized_matches: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Memilih sampel HarmBench yang tidak cocok secara ternormalisasi
    dengan AdvBench maupun JBB.

    Pemilihan ini hanya berdasarkan teks, bukan kemiripan semantik.
    """
    excluded_ids = matched_target_ids(normalized_matches)

    return [
        record
        for record in harmbench
        if record["sample_id"] not in excluded_ids
    ]


def write_selected_csv(
    output_path: Path,
    selected: List[Dict[str, str]],
) -> None:
    """
    Menulis kandidat HarmBench yang lolos pemeriksaan teks.
    """
    fieldnames = ["sample_id", "prompt"]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in selected:
            writer.writerow({
                "sample_id": record["sample_id"],
                "prompt": record["prompt"],
            })


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("HARM BENCH OVERLAP CHECK: ADV BENCH + JBB")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load reference datasets
    reference_records = {}

    for dataset_name in REFERENCE_DATASETS:
        print(f"\n[1] Memuat {dataset_name}...")
        samples = load_samples(dataset_name)
        records = validate_samples(samples, dataset_name)

        reference_records[dataset_name] = records
        print(f"Jumlah sampel valid: {len(records)}")

    # 2. Load target dataset
    print(f"\n[2] Memuat {TARGET_DATASET}...")
    harmbench_samples = load_samples(TARGET_DATASET)
    harmbench = validate_samples(
        harmbench_samples,
        TARGET_DATASET,
    )

    print(f"Jumlah sampel HarmBench: {len(harmbench)}")

    # 3. Compare exact prompts against each reference dataset
    print("\n[3] Memeriksa kecocokan teks persis...")
    exact_matches = []

    for dataset_name in REFERENCE_DATASETS:
        matches = find_matches(
            harmbench,
            reference_records[dataset_name],
            normalized=False,
        )
        exact_matches.extend(matches)

        print(
            f"HarmBench vs {dataset_name}: "
            f"{len(matches)} pasangan cocok"
        )

    write_matches_csv(EXACT_CSV, exact_matches)

    # 4. Compare normalized prompts against each reference dataset
    print("\n[4] Memeriksa kecocokan setelah normalisasi...")
    normalized_matches = []

    for dataset_name in REFERENCE_DATASETS:
        matches = find_matches(
            harmbench,
            reference_records[dataset_name],
            normalized=True,
        )
        normalized_matches.extend(matches)

        print(
            f"HarmBench vs {dataset_name}: "
            f"{len(matches)} pasangan cocok"
        )

    write_matches_csv(NORMALIZED_CSV, normalized_matches)

    # 5. Select HarmBench samples not matched to either dataset
    print("\n[5] Menyusun kandidat seleksi HarmBench...")
    excluded_ids = matched_target_ids(normalized_matches)

    selected = select_unique_samples(
        harmbench,
        normalized_matches,
    )

    write_selected_csv(SELECTED_CSV, selected)

    # 6. Build report
    exact_ids = matched_target_ids(exact_matches)

    report = {
        "target_dataset": TARGET_DATASET,
        "reference_datasets": list(REFERENCE_DATASETS),
        "dataset_sizes": {
            name: len(records)
            for name, records in reference_records.items()
        },
        "harmbench_size": len(harmbench),
        "exact_match": {
            "matching_pairs": len(exact_matches),
            "unique_harmbench_ids_matched": len(exact_ids),
        },
        "normalized_match": {
            "matching_pairs": len(normalized_matches),
            "unique_harmbench_ids_matched": len(excluded_ids),
        },
        "selection": {
            "selected_count": len(selected),
            "excluded_count": len(excluded_ids),
            "selected_sample_ids": [
                record["sample_id"] for record in selected
            ],
            "rule": (
                "Exclude a HarmBench sample if its prompt matches "
                "an AdvBench or JBB prompt after casefold and "
                "whitespace normalization."
            ),
        },
        "limitations": [
            (
                "Text matching does not detect all semantic duplicates."
            ),
            (
                "A text match does not establish identical benchmark "
                "intent, labels, or annotation methodology."
            ),
            (
                "The selected CSV is a candidate selection, not a "
                "semantic-deduplication guarantee."
            ),
            (
                "Original benchmark data is not modified."
            ),
        ],
        "output_files": {
            "exact_matches": str(
                EXACT_CSV.relative_to(PROJECT_ROOT)
            ),
            "normalized_matches": str(
                NORMALIZED_CSV.relative_to(PROJECT_ROOT)
            ),
            "selected_harmbench": str(
                SELECTED_CSV.relative_to(PROJECT_ROOT)
            ),
            "report": str(
                REPORT_JSON.relative_to(PROJECT_ROOT)
            ),
        },
    }

    with REPORT_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    # 7. Summary
    print("\n" + "=" * 70)
    print("RINGKASAN")
    print("=" * 70)

    print(f"HarmBench awal                 : {len(harmbench)}")
    print(f"Pasangan cocok persis          : {len(exact_matches)}")
    print(f"ID HarmBench cocok persis      : {len(exact_ids)}")
    print(
        "Pasangan cocok ternormalisasi : "
        f"{len(normalized_matches)}"
    )
    print(f"ID HarmBench yang dikecualikan : {len(excluded_ids)}")
    print(f"Kandidat HarmBench terpilih    : {len(selected)}")

    print("\nFile output:")
    print(f"- {EXACT_CSV}")
    print(f"- {NORMALIZED_CSV}")
    print(f"- {SELECTED_CSV}")
    print(f"- {REPORT_JSON}")

    print("\nSelesai. Dataset asli tidak diubah.")


if __name__ == "__main__":
    main()
