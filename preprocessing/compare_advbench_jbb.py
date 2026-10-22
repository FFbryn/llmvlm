
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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

EXACT_CSV = OUTPUT_DIR / "advbench_jbb_exact_duplicates.csv"
NORMALIZED_CSV = OUTPUT_DIR / "advbench_jbb_normalized_duplicates.csv"
REPORT_JSON = OUTPUT_DIR / "advbench_jbb_overlap_report.json"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_prompt(prompt: str) -> str:
    """
    Menormalkan teks untuk mendeteksi duplikasi yang berbeda
    hanya karena kapitalisasi atau whitespace.

    Tanda baca dan urutan kata tetap dipertahankan.
    """
    return " ".join(prompt.casefold().split())


# ============================================================
# DATASET LOADING
# ============================================================

def load_samples(dataset_name: str):
    """
    Memuat dataset melalui DatasetLoader.

    Pemanggilan load() harus disesuaikan jika implementasi
    DatasetLoader pada project menggunakan nama metode berbeda.
    """
    loader = DatasetLoader(dataset_name)
    return loader.load()


def extract_sample_fields(sample) -> Tuple[str, str]:
    """
    Mengambil ID dan prompt dari sebuah sample.

    Sesuaikan nama atribut di sini jika BenchmarkSample Anda
    menggunakan nama field yang berbeda.
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
    Memastikan ID sampel unik dan setiap prompt valid.
    """
    records = []
    seen_ids = set()

    for sample in samples:
        sample_id, prompt = extract_sample_fields(sample)

        if sample_id in seen_ids:
            raise ValueError(
                f"ID sampel duplikat pada dataset {dataset_name}: "
                f"{sample_id}"
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

def find_duplicates(
    advbench: List[Dict[str, str]],
    jbb: List[Dict[str, str]],
    key_function: Callable[[str], str],
) -> List[Dict[str, str]]:
    """
    Mencocokkan prompt AdvBench dengan prompt JBB berdasarkan
    fungsi kunci yang diberikan.

    Satu prompt dapat cocok dengan beberapa sampel di dataset
    lainnya. Semua pasangan dicatat agar tidak kehilangan data.
    """
    advbench_index = defaultdict(list)

    for record in advbench:
        key = key_function(record["prompt"])

        if key:
            advbench_index[key].append(record)

    matches = []

    for jbb_record in jbb:
        key = key_function(jbb_record["prompt"])

        if not key:
            continue

        for adv_record in advbench_index.get(key, []):
            matches.append({
                "advbench_sample_id": adv_record["sample_id"],
                "jbb_sample_id": jbb_record["sample_id"],
                "advbench_prompt": adv_record["prompt"],
                "jbb_prompt": jbb_record["prompt"],
            })

    return matches


def write_matches_csv(
    output_path: Path,
    matches: List[Dict[str, str]],
) -> None:
    """
    Menulis pasangan sampel yang cocok ke CSV.
    """
    fieldnames = [
        "advbench_sample_id",
        "jbb_sample_id",
        "advbench_prompt",
        "jbb_prompt",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)


def count_matched_ids(
    matches: List[Dict[str, str]],
) -> Tuple[int, int]:
    """
    Menghitung jumlah ID unik yang terlibat dalam pasangan cocok.
    """
    advbench_ids = {
        row["advbench_sample_id"] for row in matches
    }
    jbb_ids = {
        row["jbb_sample_id"] for row in matches
    }

    return len(advbench_ids), len(jbb_ids)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 65)
    print("ADVBench vs JBB DATASET OVERLAP CHECK")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] Memuat AdvBench...")
    advbench_samples = load_samples("advbench")
    advbench = validate_samples(advbench_samples, "advbench")
    print(f"AdvBench valid: {len(advbench)} sampel")

    print("\n[2] Memuat JBB...")
    jbb_samples = load_samples("jbb")
    jbb = validate_samples(jbb_samples, "jbb")
    print(f"JBB valid: {len(jbb)} sampel")

    print("\n[3] Memeriksa duplikasi teks persis...")
    exact_matches = find_duplicates(
        advbench,
        jbb,
        key_function=lambda prompt: prompt,
    )
    write_matches_csv(EXACT_CSV, exact_matches)

    print(f"Pasangan cocok persis: {len(exact_matches)}")

    print("\n[4] Memeriksa duplikasi setelah normalisasi...")
    normalized_matches = find_duplicates(
        advbench,
        jbb,
        key_function=normalize_prompt,
    )
    write_matches_csv(NORMALIZED_CSV, normalized_matches)

    print(
        "Pasangan cocok setelah normalisasi: "
        f"{len(normalized_matches)}"
    )

    exact_adv_ids, exact_jbb_ids = count_matched_ids(exact_matches)
    norm_adv_ids, norm_jbb_ids = count_matched_ids(normalized_matches)

    report = {
        "comparison": "AdvBench vs JBB",
        "method": {
            "exact": (
                "Mencocokkan prompt berdasarkan teks persis, "
                "tanpa normalisasi."
            ),
            "normalized": (
                "casefold dan penggabungan whitespace; "
                "tanda baca tetap dipertahankan."
            ),
        },
        "dataset_sizes": {
            "advbench": len(advbench),
            "jbb": len(jbb),
        },
        "exact_match": {
            "matching_pairs": len(exact_matches),
            "unique_advbench_ids_matched": exact_adv_ids,
            "unique_jbb_ids_matched": exact_jbb_ids,
            "advbench_ids_without_match": (
                len(advbench) - exact_adv_ids
            ),
            "jbb_ids_without_match": (
                len(jbb) - exact_jbb_ids
            ),
        },
        "normalized_match": {
            "matching_pairs": len(normalized_matches),
            "unique_advbench_ids_matched": norm_adv_ids,
            "unique_jbb_ids_matched": norm_jbb_ids,
            "advbench_ids_without_match": (
                len(advbench) - norm_adv_ids
            ),
            "jbb_ids_without_match": (
                len(jbb) - norm_jbb_ids
            ),
        },
        "limitations": [
            (
                "Pencocokan ini tidak mendeteksi seluruh kemiripan "
                "semantik antar-prompt."
            ),
            (
                "Kecocokan prompt tidak membuktikan bahwa kedua "
                "benchmark identik dalam tujuan atau anotasinya."
            ),
            (
                "Pasangan hasil pencocokan tidak otomatis dihapus "
                "dari dataset eksperimen."
            ),
        ],
        "output_files": {
            "exact_matches": str(EXACT_CSV.relative_to(PROJECT_ROOT)),
            "normalized_matches": str(
                NORMALIZED_CSV.relative_to(PROJECT_ROOT)
            ),
            "report": str(REPORT_JSON.relative_to(PROJECT_ROOT)),
        },
    }

    with REPORT_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n[5] Ringkasan")
    print("-" * 65)
    print(f"Jumlah AdvBench                : {len(advbench)}")
    print(f"Jumlah JBB                     : {len(jbb)}")
    print(f"Pasangan cocok persis          : {len(exact_matches)}")
    print(f"ID AdvBench cocok persis       : {exact_adv_ids}")
    print(f"ID JBB cocok persis            : {exact_jbb_ids}")
    print(f"Pasangan cocok ternormalisasi  : {len(normalized_matches)}")
    print(f"ID AdvBench ternormalisasi     : {norm_adv_ids}")
    print(f"ID JBB ternormalisasi          : {norm_jbb_ids}")

    print("\nFile output:")
    print(f"- {EXACT_CSV}")
    print(f"- {NORMALIZED_CSV}")
    print(f"- {REPORT_JSON}")

    print("\nPemeriksaan overlap selesai.")
    print(
        "Catatan: hasil ini hanya laporan perbandingan, "
        "bukan penghapusan sampel."
    )


if __name__ == "__main__":
    main()
