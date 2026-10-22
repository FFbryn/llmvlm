
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

RESULTS_DIR = PROJECT_ROOT / "benchmark_data" / "results"
OVERLAP_DIR = RESULTS_DIR / "dataset_overlap"

HARMBENCH_SELECTED_CSV = (
    RESULTS_DIR
    / "dataset_overlap"
    / "harmbench_selected_unique.csv"
)

EXACT_CSV = OVERLAP_DIR / "xstest_existing_exact_duplicates.csv"
NORMALIZED_CSV = OVERLAP_DIR / "xstest_existing_normalized_duplicates.csv"
SELECTED_CSV = OVERLAP_DIR / "xstest_selected_unique.csv"
REPORT_JSON = OVERLAP_DIR / "xstest_overlap_report.json"

TARGET_DATASET = "xstest"
REFERENCE_DATASETS = ("advbench", "jbb")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_prompt(prompt: str) -> str:
    """
    Normalisasi untuk mendeteksi perbedaan kapitalisasi
    dan whitespace.

    Tanda baca dan urutan kata tidak diubah.
    """
    return " ".join(prompt.casefold().split())


# ============================================================
# LOADING AND VALIDATION
# ============================================================

def load_dataset(dataset_name: str):
    """
    Memuat dataset melalui DatasetLoader.

    Jika API loader project berbeda, sesuaikan fungsi ini
    dengan implementasi benchmark_data/loader.py.
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
            f"Prompt kosong atau tidak valid pada sample {sample_id!r}."
        )

    return str(sample_id), prompt


def validate_samples(samples, dataset_name: str) -> List[Dict[str, str]]:
    """
    Memastikan setiap sample memiliki ID unik dan prompt valid.
    """
    records = []
    seen_ids = set()

    for sample in samples:
        sample_id, prompt = extract_sample_fields(sample)

        if sample_id in seen_ids:
            raise ValueError(
                f"ID sampel duplikat dalam {dataset_name}: {sample_id}"
            )

        seen_ids.add(sample_id)

        records.append({
            "dataset": dataset_name,
            "sample_id": sample_id,
            "prompt": prompt,
        })

    return records


def load_selected_harmbench(path: Path) -> List[Dict[str, str]]:
    """
    Membaca kandidat HarmBench dari hasil seleksi tahap sebelumnya.

    File yang diharapkan memiliki kolom sample_id dan prompt.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"File seleksi HarmBench tidak ditemukan:\n{path}\n"
            "Jalankan preprocessing.compare_harmbench_existing terlebih "
            "dahulu, lalu periksa hasilnya."
        )

    records = []
    seen_ids = set()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"sample_id", "prompt"}
        actual_columns = set(reader.fieldnames or [])

        if not required_columns.issubset(actual_columns):
            raise ValueError(
                f"Kolom file HarmBench tidak sesuai. "
                f"Diperlukan: {sorted(required_columns)}; "
                f"tersedia: {sorted(actual_columns)}"
            )

        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            prompt = row.get("prompt") or ""

            if not sample_id:
                raise ValueError(
                    "Ditemukan baris HarmBench dengan sample_id kosong."
                )

            if not prompt.strip():
                raise ValueError(
                    f"Prompt HarmBench kosong untuk ID {sample_id!r}."
                )

            if sample_id in seen_ids:
                raise ValueError(
                    f"ID duplikat dalam file seleksi HarmBench: {sample_id}"
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
    """
    Membandingkan XSTest dengan satu dataset referensi.

    Semua pasangan cocok dicatat. Satu sampel bisa cocok dengan
    lebih dari satu sampel pada dataset referensi.
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
    path: Path,
    matches: List[Dict[str, str]],
) -> None:
    """
    Menyimpan pasangan yang cocok ke CSV.
    """
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


def get_matched_target_ids(
    matches: List[Dict[str, str]],
) -> set:
    """
    Mengambil ID XSTest unik yang cocok dengan dataset referensi.
    """
    return {
        row["target_sample_id"]
        for row in matches
    }


# ============================================================
# SELECTION
# ============================================================

def select_unique_samples(
    xstest: List[Dict[str, str]],
    normalized_matches: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Mengecualikan XSTest yang memiliki kecocokan teks ternormalisasi
    dengan AdvBench, JBB, atau kandidat HarmBench terpilih.
    """
    excluded_ids = get_matched_target_ids(normalized_matches)

    return [
        record
        for record in xstest
        if record["sample_id"] not in excluded_ids
    ]


def write_selected_csv(
    path: Path,
    selected: List[Dict[str, str]],
) -> None:
    """
    Menulis kandidat XSTest ke CSV tanpa mengubah dataset sumber.
    """
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "prompt"],
        )
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
    print("XSTEST OVERLAP CHECK: ADV BENCH + JBB + HARM BENCH")
    print("=" * 70)

    OVERLAP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load AdvBench and JBB
    references = {}

    for dataset_name in REFERENCE_DATASETS:
        print(f"\n[1] Memuat {dataset_name}...")
        samples = load_dataset(dataset_name)
        references[dataset_name] = validate_samples(
            samples,
            dataset_name,
        )
        print(
            f"Jumlah sampel valid: {len(references[dataset_name])}"
        )

    # 2. Load selected HarmBench
    print("\n[2] Membaca kandidat HarmBench terpilih...")
    references["harmbench"] = load_selected_harmbench(
    HARMBENCH_SELECTED_CSV
    )
    print(
        "Jumlah kandidat HarmBench: "
        f"{len(references['harmbench'])}"
    )

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
        matches = find_matches(
            xstest,
            records,
            normalized=False,
        )
        exact_matches.extend(matches)
        print(f"XSTest vs {dataset_name}: {len(matches)} pasangan")

    write_matches_csv(EXACT_CSV, exact_matches)

    # 5. Normalized matching
    print("\n[5] Memeriksa kecocokan setelah normalisasi...")
    normalized_matches = []

    for dataset_name, records in references.items():
        matches = find_matches(
            xstest,
            records,
            normalized=True,
        )
        normalized_matches.extend(matches)
        print(f"XSTest vs {dataset_name}: {len(matches)} pasangan")

    write_matches_csv(NORMALIZED_CSV, normalized_matches)

    # 6. Select unmatched XSTest candidates
    print("\n[6] Membuat kandidat seleksi XSTest...")
    excluded_ids = get_matched_target_ids(normalized_matches)

    selected = select_unique_samples(
        xstest,
        normalized_matches,
    )

    write_selected_csv(SELECTED_CSV, selected)

    exact_ids = get_matched_target_ids(exact_matches)

    # 7. Report
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
        },
        "limitations": [
            "Text matching does not detect all semantic duplicates.",
            (
                "The candidate list does not preserve every field "
                "or annotation from the original XSTest adapter."
            ),
            (
                "XSTest has a different evaluation purpose from "
                "harmfulness-focused benchmarks; its original labels "
                "must be preserved for subsequent experiments."
            ),
            "Original benchmark files are not modified.",
        ],
        "output_files": {
            "exact_matches": str(EXACT_CSV.relative_to(PROJECT_ROOT)),
            "normalized_matches": str(
                NORMALIZED_CSV.relative_to(PROJECT_ROOT)
            ),
            "selected_xstest": str(
                SELECTED_CSV.relative_to(PROJECT_ROOT)
            ),
            "report": str(REPORT_JSON.relative_to(PROJECT_ROOT)),
        },
    }

    with REPORT_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    # 8. Summary
    print("\n" + "=" * 70)
    print("RINGKASAN")
    print("=" * 70)
    print(f"Jumlah XSTest awal                : {len(xstest)}")
    print(f"Pasangan cocok persis             : {len(exact_matches)}")
    print(f"ID XSTest cocok persis            : {len(exact_ids)}")
    print(
        "Pasangan cocok ternormalisasi    : "
        f"{len(normalized_matches)}"
    )
    print(f"ID XSTest yang dikecualikan       : {len(excluded_ids)}")
    print(f"Kandidat XSTest terpilih          : {len(selected)}")

    print("\nFile output:")
    print(f"- {EXACT_CSV}")
    print(f"- {NORMALIZED_CSV}")
    print(f"- {SELECTED_CSV}")
    print(f"- {REPORT_JSON}")

    print("\nPemeriksaan selesai. Dataset asli tidak diubah.")


if __name__ == "__main__":
    main()
