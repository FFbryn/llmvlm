
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

OVERLAP_DIR = (
    PROJECT_ROOT / "benchmark_data" / "results" / "dataset_overlap"
)

HARMBENCH_SELECTED_CSV = OVERLAP_DIR / "harmbench_selected_unique.csv"

EXACT_CSV = OVERLAP_DIR / "xstest_existing_exact_duplicates.csv"
NORMALIZED_CSV = OVERLAP_DIR / "xstest_existing_normalized_duplicates.csv"
SELECTED_CSV = OVERLAP_DIR / "xstest_selected_unique.csv"
SELECTED_JSONL = OVERLAP_DIR / "xstest_selected_unique.jsonl"
REPORT_JSON = OVERLAP_DIR / "xstest_overlap_report.json"

TARGET_DATASET = "xstest"
REFERENCE_DATASETS = ("advbench", "jbb")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_prompt(prompt: str) -> str:
    """Normalisasi kapitalisasi dan whitespace; tanda baca dipertahankan."""
    return " ".join(prompt.casefold().split())


# ============================================================
# LOADING AND VALIDATION
# ============================================================

def load_dataset(dataset_name: str):
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
    records = []
    seen_ids = set()

    for sample in samples:
        sample_id, prompt = extract_sample_fields(sample)

        if sample_id in seen_ids:
            raise ValueError(
                f"ID duplikat dalam dataset {dataset_name}: {sample_id}"
            )

        seen_ids.add(sample_id)
        records.append({
            "dataset": dataset_name,
            "sample_id": sample_id,
            "prompt": prompt,
        })

    return records


def load_selected_harmbench(path: Path) -> List[Dict[str, str]]:
    """Membaca kandidat HarmBench dari tahap overlap sebelumnya."""
    if not path.exists():
        raise FileNotFoundError(
            f"File kandidat HarmBench tidak ditemukan:\n{path}\n"
            "Jalankan compare_harmbench terlebih dahulu."
        )

    records = []
    seen_ids = set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"sample_id", "prompt"}
        available = set(reader.fieldnames or [])

        if not required.issubset(available):
            raise ValueError(
                f"Kolom HarmBench harus mencakup {sorted(required)}; "
                f"kolom tersedia: {sorted(available)}"
            )

        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            prompt = row.get("prompt") or ""

            if not sample_id:
                raise ValueError("Ditemukan sample_id HarmBench kosong.")

            if not prompt.strip():
                raise ValueError(
                    f"Prompt HarmBench kosong untuk ID {sample_id!r}."
                )

            if sample_id in seen_ids:
                raise ValueError(
                    f"ID duplikat pada kandidat HarmBench: {sample_id}"
                )

            seen_ids.add(sample_id)
            records.append({
                "dataset": "harmbench",
                "sample_id": sample_id,
                "prompt": prompt,
            })

    return records


# ============================================================
# MATCHING
# ============================================================

def find_matches(
    target_records: List[Dict[str, str]],
    reference_records: List[Dict[str, str]],
    normalized: bool,
) -> List[Dict[str, str]]:
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
    path: Path,
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

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)


def get_matched_target_ids(matches: List[Dict[str, str]]) -> set:
    return {row["target_sample_id"] for row in matches}


# ============================================================
# SELECTION AND EXPORT
# ============================================================

def select_unique_samples(
    xstest: List[Dict[str, str]],
    normalized_matches: List[Dict[str, str]],
):
    excluded_ids = get_matched_target_ids(normalized_matches)

    selected = [
        record
        for record in xstest
        if record["sample_id"] not in excluded_ids
    ]

    return selected, excluded_ids


def write_selected_files(
    selected: List[Dict[str, str]],
) -> None:
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
            file.write(json.dumps({
                "sample_id": record["sample_id"],
                "prompt": record["prompt"],
            }, ensure_ascii=False) + "\n")


def validate_selection(
    original: List[Dict[str, str]],
    selected: List[Dict[str, str]],
    excluded_ids: set,
) -> Dict[str, bool]:
    original_ids = {record["sample_id"] for record in original}
    selected_ids = [record["sample_id"] for record in selected]
    selected_set = set(selected_ids)

    checks = {
        "selected_ids_unique": len(selected_ids) == len(selected_set),
        "excluded_ids_absent": not bool(selected_set & excluded_ids),
        "selected_ids_from_original": selected_set.issubset(original_ids),
        "counts_consistent": (
            len(selected) + len(excluded_ids) == len(original)
        ),
        "all_original_ids_accounted_for": (
            selected_set | excluded_ids == original_ids
        ),
    }

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "Verifikasi seleksi gagal: " + ", ".join(failed)
        )

    return checks


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("XSTEST OVERLAP CHECK + CANDIDATE EXPORT")
    print("=" * 70)

    OVERLAP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load AdvBench and JBB
    references = {}

    for dataset_name in REFERENCE_DATASETS:
        print(f"\n[1] Memuat {dataset_name}...")
        references[dataset_name] = validate_samples(
            load_dataset(dataset_name),
            dataset_name,
        )
        print(f"Jumlah sampel valid: {len(references[dataset_name])}")

    # 2. Load previously filtered HarmBench
    print("\n[2] Memuat kandidat HarmBench...")
    references["harmbench"] = load_selected_harmbench(
        HARMBENCH_SELECTED_CSV
    )
    print(f"Jumlah kandidat HarmBench: {len(references['harmbench'])}")

    # 3. Load XSTest
    print("\n[3] Memuat XSTest...")
    xstest = validate_samples(
        load_dataset(TARGET_DATASET),
        TARGET_DATASET,
    )
    print(f"Jumlah sampel XSTest: {len(xstest)}")

    # 4. Exact matching
    print("\n[4] Memeriksa kecocokan teks persis...")
    exact_matches = []

    for dataset_name, records in references.items():
        matches = find_matches(xstest, records, normalized=False)
        exact_matches.extend(matches)
        print(f"XSTest vs {dataset_name}: {len(matches)} pasangan")

    write_matches_csv(EXACT_CSV, exact_matches)

    # 5. Normalized matching
    print("\n[5] Memeriksa kecocokan setelah normalisasi...")
    normalized_matches = []

    for dataset_name, records in references.items():
        matches = find_matches(xstest, records, normalized=True)
        normalized_matches.extend(matches)
        print(f"XSTest vs {dataset_name}: {len(matches)} pasangan")

    write_matches_csv(NORMALIZED_CSV, normalized_matches)

    # 6. Select and validate
    print("\n[6] Membuat kandidat XSTest...")
    selected, excluded_ids = select_unique_samples(
        xstest,
        normalized_matches,
    )
    write_selected_files(selected)
    checks = validate_selection(xstest, selected, excluded_ids)

    exact_ids = get_matched_target_ids(exact_matches)

    print(f"XSTest awal                    : {len(xstest)}")
    print(f"ID XSTest cocok persis          : {len(exact_ids)}")
    print(f"ID XSTest dikeluarkan           : {len(excluded_ids)}")
    print(f"Kandidat XSTest tersisa         : {len(selected)}")
    print("Verifikasi integritas           : LULUS")

    # 7. Save report
    report = {
        "target_dataset": TARGET_DATASET,
        "reference_datasets": list(references.keys()),
        "dataset_sizes": {
            name: len(records)
            for name, records in references.items()
        },
        "xstest_size": len(xstest),
        "exact_match": {
            "matching_pairs": len(exact_matches),
            "unique_xstest_ids_matched": len(exact_ids),
        },
        "normalized_match": {
            "matching_pairs": len(normalized_matches),
            "unique_xstest_ids_matched": len(excluded_ids),
        },
        "selection": {
            "selected_count": len(selected),
            "excluded_count": len(excluded_ids),
            "selected_sample_ids": [
                record["sample_id"] for record in selected
            ],
            "rule": (
                "Exclude an XSTest sample if its prompt matches "
                "AdvBench, JBB, or selected HarmBench after "
                "casefold and whitespace normalization."
            ),
            "validation": checks,
        },
        "limitations": [
            "Text matching does not detect all semantic duplicates.",
            (
                "This export retains sample IDs and prompts only; "
                "other original XSTest annotations are not included."
            ),
            (
                "XSTest has a different evaluation purpose from "
                "harmfulness-focused benchmarks. Preserve its original "
                "labels for downstream refusal and over-refusal analysis."
            ),
            "Original benchmark files are not modified.",
        ],
        "output_files": {
            "exact_matches": str(EXACT_CSV.relative_to(PROJECT_ROOT)),
            "normalized_matches": str(
                NORMALIZED_CSV.relative_to(PROJECT_ROOT)
            ),
            "selected_csv": str(SELECTED_CSV.relative_to(PROJECT_ROOT)),
            "selected_jsonl": str(
                SELECTED_JSONL.relative_to(PROJECT_ROOT)
            ),
            "report": str(REPORT_JSON.relative_to(PROJECT_ROOT)),
        },
    }

    with REPORT_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nFile output:")
    print(f"- {EXACT_CSV}")
    print(f"- {NORMALIZED_CSV}")
    print(f"- {SELECTED_CSV}")
    print(f"- {SELECTED_JSONL}")
    print(f"- {REPORT_JSON}")

    print("\nSelesai. Dataset sumber tidak diubah.")


if __name__ == "__main__":
    main()
