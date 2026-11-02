from collections import Counter
import json
from pathlib import Path

from evaluation.classification.classification_runner import ClassificationRunner
from evaluation.classification.labels import ClassificationLabel
from evaluation.classification.rule_based import RuleBasedResponseClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "real_batch_100"
)

CLASSIFIED_RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "classified_100"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

EXPECTED_SAMPLE_COUNT = 100

CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)

DATASETS = {
    "advbench": {
        "input": "advbench_100_real_batch.jsonl",
        "output": "advbench_100_classified.jsonl",
    },
    "harmbench": {
        "input": "harmbench_100_real_batch.jsonl",
        "output": "harmbench_100_classified.jsonl",
    },
    "xstest": {
        "input": "xstest_100_real_batch.jsonl",
        "output": "xstest_100_classified.jsonl",
    },
}

VALID_LABELS = {
    label.value for label in ClassificationLabel
}


# ============================================================
# JSONL HELPERS
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL and report the exact line if parsing fails."""

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada {path}, "
                    f"baris {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record pada {path}, baris {line_number} "
                    "bukan JSON object."
                )

            records.append(record)

    return records


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_raw_input(path: Path) -> list[dict]:
    """Validate sample count, IDs, conditions, and responses."""

    if not path.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan:\n{path}"
        )

    records = load_jsonl(path)

    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Jumlah record tidak sesuai pada {path.name}. "
            f"Diharapkan {EXPECTED_SAMPLE_COUNT}, "
            f"ditemukan {len(records)}."
        )

    sample_ids = []

    for index, record in enumerate(records, start=1):
        sample_id = record.get("sample_id")

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(
                f"sample_id kosong/tidak valid pada record {index} "
                f"di {path.name}."
            )

        sample_ids.append(sample_id)

        for condition in CONDITIONS:
            condition_data = record.get(condition)

            if not isinstance(condition_data, dict):
                raise ValueError(
                    f"Condition '{condition}' tidak ditemukan "
                    f"atau bukan object pada sample_id={sample_id}."
                )

            response = condition_data.get("response")

            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    f"Response kosong pada sample_id={sample_id}, "
                    f"condition={condition}."
                )

    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(
            f"Ditemukan sample_id duplikat pada {path.name}."
        )

    print(f"  Validasi input: OK ({len(records)} sampel)")
    print(f"  Kondisi per sampel: {', '.join(CONDITIONS)}")

    return records


# ============================================================
# CLASSIFIED OUTPUT VALIDATION
# ============================================================

def validate_classified_output(
    path: Path,
    expected_records: list[dict],
) -> dict:
    """Validate classified records and summarize labels."""

    if not path.exists():
        raise FileNotFoundError(
            f"Output klasifikasi tidak terbentuk:\n{path}"
        )

    records = load_jsonl(path)

    if len(records) != len(expected_records):
        raise ValueError(
            f"Jumlah record output salah pada {path.name}. "
            f"Input={len(expected_records)}, output={len(records)}."
        )

    expected_ids = [
        record["sample_id"] for record in expected_records
    ]
    actual_ids = [
        record.get("sample_id") for record in records
    ]

    if actual_ids != expected_ids:
        raise ValueError(
            f"sample_id output tidak sesuai urutan input "
            f"pada {path.name}."
        )

    counts = {
        condition: Counter()
        for condition in CONDITIONS
    }

    for record in records:
        sample_id = record["sample_id"]

        for condition in CONDITIONS:
            condition_data = record.get(condition)

            if not isinstance(condition_data, dict):
                raise ValueError(
                    f"Condition '{condition}' hilang pada "
                    f"sample_id={sample_id} di output."
                )

            label = condition_data.get("classification")

            if label not in VALID_LABELS:
                raise ValueError(
                    f"Label tidak valid pada sample_id={sample_id}, "
                    f"condition={condition}: {label!r}"
                )

            counts[condition][label] += 1

    for condition in CONDITIONS:
        total = sum(counts[condition].values())

        if total != EXPECTED_SAMPLE_COUNT:
            raise ValueError(
                f"Jumlah klasifikasi untuk {condition} adalah "
                f"{total}, seharusnya {EXPECTED_SAMPLE_COUNT}."
            )

    return counts


# ============================================================
# CLASSIFY ONE DATASET
# ============================================================

def classify_dataset(
    dataset_name: str,
    input_filename: str,
    output_filename: str,
) -> bool:
    """Run classification for one dataset."""

    input_path = RAW_RESULTS_DIR / input_filename
    output_path = CLASSIFIED_RESULTS_DIR / output_filename

    print()
    print("=" * 72)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 72)

    try:
        # 1. Validate raw batch results before classification.
        raw_records = validate_raw_input(input_path)

        # 2. Prepare output directory.
        CLASSIFIED_RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 3. Remove previous output because the writer appends
        #    records instead of replacing the file.
        if output_path.exists():
            output_path.unlink()
            print("  Output lama dihapus agar tidak terjadi duplikasi.")

        # 4. Initialize the existing rule-based classifier.
        classifier = RuleBasedResponseClassifier()

        # 5. Run classification.
        runner = ClassificationRunner(
            classifier=classifier,
            input_path=input_path,
            output_path=output_path,
        )

        processed_count = runner.run()

        if processed_count != EXPECTED_SAMPLE_COUNT:
            raise ValueError(
                f"Runner memproses {processed_count} record; "
                f"diharapkan {EXPECTED_SAMPLE_COUNT}."
            )

        # 6. Validate the resulting classified file.
        counts = validate_classified_output(
            output_path,
            raw_records,
        )

        # 7. Print label distributions for each condition.
        print("\nDistribusi label:")

        for condition in CONDITIONS:
            print(f"\n  {condition}:")

            for label in sorted(VALID_LABELS):
                print(
                    f"    {label:<12}: "
                    f"{counts[condition].get(label, 0):>3}"
                )

        print(f"\n  Output: {output_path}")
        print("  Status: BERHASIL")

        return True

    except Exception as exc:
        print(f"\n  Status: GAGAL")
        print(f"  Penyebab: {exc}")
        return False


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 72)
    print("KLASIFIKASI HASIL BATCH: ADVBench, HarmBench, XSTest")
    print("=" * 72)

    print(f"Input directory : {RAW_RESULTS_DIR}")
    print(f"Output directory: {CLASSIFIED_RESULTS_DIR}")
    print(f"Target sampel   : {EXPECTED_SAMPLE_COUNT} per dataset")

    results = {}

    for dataset_name, config in DATASETS.items():
        results[dataset_name] = classify_dataset(
            dataset_name=dataset_name,
            input_filename=config["input"],
            output_filename=config["output"],
        )

    print("\n")
    print("=" * 72)
    print("RINGKASAN AKHIR")
    print("=" * 72)

    for dataset_name, success in results.items():
        status = "BERHASIL" if success else "GAGAL"
        print(f"{dataset_name:<12}: {status}")

    successful = sum(results.values())
    print(
        f"\nDataset berhasil: {successful}/{len(DATASETS)}"
    )

    if successful != len(DATASETS):
        print(
            "\nPERHATIAN: Ada dataset yang gagal. "
            "Periksa pesan error di atas sebelum melanjutkan "
            "ke export ambiguous."
        )
    else:
        print(
            "\nSemua dataset selesai diklasifikasikan dan "
            "validasi output berhasil."
        )


if __name__ == "__main__":
    main()