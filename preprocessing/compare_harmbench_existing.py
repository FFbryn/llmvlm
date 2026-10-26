
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
SELECTED_JSONL = OUTPUT_DIR / "harmbench_selected_unique.jsonl"
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
    """Normalisasi kapitalisasi dan whitespace; tanda baca dipertahankan."""
    return " ".join(prompt.casefold().split())


# ============================================================
# DATASET LOADING AND VALIDATION
# ============================================================

def load_samples(dataset_name: str):
    """Memuat dataset melalui DatasetLoader."""
    return DatasetLoader(dataset_name).load()


def extract_sample_fields(sample) -> Tuple[str, str]:
    sample_id = getattr(sample, "sample_id", None)
    prompt = getattr(sample, "prompt", None)

    if sample_id is None:
        raise ValueError(
            "Sample tidak memiliki atribut 'sample_id'. "
            "Periksa schema dan implementasi DatasetLoader."
        )

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"Prompt kosong atau tidak valid pada sampel {sample_id!r}."
        )

    return str(sample_id), prompt


def validate_samples(samples, dataset_name: str) -> List[Dict[str, str]]:
    """Validasi ID unik dan prompt tidak kosong."""
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
    Mencatat semua pasangan target dan referensi yang cocok.
    Pencocokan tidak mengukur kemiripan semantik.
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


def matched_target_ids(matches: List[Dict[str, str]]) -> set:
    """Mengembalikan ID target unik yang memiliki pasangan cocok."""
    return {row["target_sample_id"] for row in matches}


# ============================================================
# CANDIDATE SELECTION AND EXPORT
# ============================================================

def select_unique_samples(
    harmbench: List[Dict[str, str]],
    normalized_matches: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], set]:
    """
    Mengeluarkan HarmBench yang cocok secara ternormalisasi
    dengan AdvBench atau JBB.
    """
    excluded_ids = matched_target_ids(normalized_matches)

    selected = [
        record
        for record in harmbench
        if record["sample_id"] not in excluded_ids
    ]

    return selected, excluded_ids


def write_selected_files(
    selected: List[Dict[str, str]],
) -> None:
    """Menyimpan kandidat dalam CSV dan JSONL."""
    fieldnames = ["sample_id", "prompt"]

    with SELECTED_CSV.open(
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

    with SELECTED_JSONL.open("w", encoding="utf-8") as file:
        for record in selected:
            output_record = {
                "sample_id": record["sample_id"],
                "prompt": record["prompt"],
            }
            file.write(
                json.dumps(output_record, ensure_ascii=False) + "\n"
            )


def validate_selection(
    harmbench: List[Dict[str, str]],
    selected: List[Dict[str, str]],
    excluded_ids: set,
) -> Dict[str, bool]:
    """Memeriksa konsistensi jumlah dan ID kandidat."""
    original_ids = {record["sample_id"] for record in harmbench}
    selected_ids = [record["sample_id"] for record in selected]
    selected_id_set = set(selected_ids)

    checks = {
        "selected_ids_unique": (
            len(selected_ids) == len(selected_id_set)
        ),
        "excluded_ids_absent": (
            not bool(selected_id_set & excluded_ids)
        ),
        "selected_ids_from_original": (
            selected_id_set.issubset(original_ids)
        ),
        "counts_consistent": (
            len(selected) + len(excluded_ids) == len(harmbench)
        ),
        "original_ids_accounted_for": (
            selected_id_set | excluded_ids == original_ids
        ),
    }

    failed = [name for name, passed in checks.items() if not passed]

    if failed:
        raise RuntimeError(
            "Verifikasi seleksi gagal pada: " + ", ".join(failed)
        )

    return checks


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("HARM BENCH OVERLAP CHECK + CANDIDATE EXPORT")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load reference datasets
    reference_records = {}

    for dataset_name in REFERENCE_DATASETS:
        print(f"\n[1] Memuat {dataset_name}...")
        records = validate_samples(
            load_samples(dataset_name),
            dataset_name,
        )
        reference_records[dataset_name] = records
        print(f"Jumlah sampel valid: {len(records)}")

    # 2. Load target dataset
    print(f"\n[2] Memuat {TARGET_DATASET}...")
    harmbench = validate_samples(
        load_samples(TARGET_DATASET),
        TARGET_DATASET,
    )
    print(f"Jumlah sampel HarmBench: {len(harmbench)}")

    # 3. Exact matching
    print("\n[3] Memeriksa kecocokan teks persis...")
    exact_matches = []

    for dataset_name in REFERENCE_DATASETS:
        matches = find_matches(
            harmbench,
            reference_records[dataset_name],
            normalized=False,
        )
        exact_matches.extend(matches)
        print(f"HarmBench vs {dataset_name}: {len(matches)} pasangan")

    write_matches_csv(EXACT_CSV, exact_matches)

    # 4. Normalized matching
    print("\n[4] Memeriksa kecocokan setelah normalisasi...")
    normalized_matches = []

    for dataset_name in REFERENCE_DATASETS:
        matches = find_matches(
            harmbench,
            reference_records[dataset_name],
            normalized=True,
        )
        normalized_matches.extend(matches)
        print(f"HarmBench vs {dataset_name}: {len(matches)} pasangan")

    write_matches_csv(NORMALIZED_CSV, normalized_matches)

    # 5. Remove matched target samples and export candidates
    print("\n[5] Membuat dataset kandidat HarmBench...")
    selected, excluded_ids = select_unique_samples(
        harmbench,
        normalized_matches,
    )

    write_selected_files(selected)
    checks = validate_selection(harmbench, selected, excluded_ids)

    print(f"HarmBench awal                 : {len(harmbench)}")
    print(f"ID HarmBench dikeluarkan       : {len(excluded_ids)}")
    print(f"Kandidat HarmBench tersisa     : {len(selected)}")
    print("Verifikasi integritas          : LULUS")

    # 6. Report
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
            "validation": checks,
        },
        "limitations": [
            "Text matching does not detect all semantic duplicates.",
            (
                "A text match does not establish identical benchmark "
                "intent, labels, or annotation methodology."
            ),
            "Original benchmark data is not modified.",
        ],
        "output_files": {
            "exact_matches": str(EXACT_CSV.relative_to(PROJECT_ROOT)),
            "normalized_matches": str(
                NORMALIZED_CSV.relative_to(PROJECT_ROOT)
            ),
            "selected_harmbench_csv": str(
                SELECTED_CSV.relative_to(PROJECT_ROOT)
            ),
            "selected_harmbench_jsonl": str(
                SELECTED_JSONL.relative_to(PROJECT_ROOT)
            ),
            "report": str(REPORT_JSON.relative_to(PROJECT_ROOT)),
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
    print(f"Pasangan cocok ternormalisasi  : {len(normalized_matches)}")
    print(f"ID HarmBench dikecualikan      : {len(excluded_ids)}")
    print(f"Kandidat HarmBench tersisa     : {len(selected)}")

    print("\nFile output:")
    print(f"- {EXACT_CSV}")
    print(f"- {NORMALIZED_CSV}")
    print(f"- {SELECTED_CSV}")
    print(f"- {SELECTED_JSONL}")
    print(f"- {REPORT_JSON}")

    print("\nSelesai. Dataset sumber tidak diubah.")


if __name__ == "__main__":
    main()
