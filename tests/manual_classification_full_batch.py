
import json
from collections import Counter
from pathlib import Path

from evaluation.classification import (
    ClassificationRunner,
    RuleBasedResponseClassifier,
)
from evaluation.classification.labels import ClassificationLabel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "jbb_100"
)

INPUT_PATH = RESULTS_DIR / "jbb_100_raw.jsonl"
OUTPUT_PATH = RESULTS_DIR / "jbb_100_classified.jsonl"

EXPECTED_SAMPLE_COUNT = 100

CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)

VALID_LABELS = {
    label.value for label in ClassificationLabel
}


def read_jsonl(path: Path) -> list[dict]:
    """Baca JSONL dan laporkan lokasi jika ada JSON yang rusak."""
    if not path.is_file():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

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
                    f"Record baris {line_number} bukan object JSON."
                )

            records.append(record)

    return records


def validate_raw_records(records: list[dict]) -> list[str]:
    """Validasi ID dan respons untuk ketiga kondisi eksperimen."""
    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Diharapkan {EXPECTED_SAMPLE_COUNT} sampel, "
            f"tetapi ditemukan {len(records)}."
        )

    sample_ids = []

    for index, record in enumerate(records, start=1):
        sample_id = record.get("sample_id")

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(
                f"Record ke-{index} tidak memiliki sample_id valid."
            )

        sample_ids.append(sample_id)

        for condition_name in CONDITIONS:
            condition = record.get(condition_name)

            if not isinstance(condition, dict):
                raise ValueError(
                    f"{sample_id}: kondisi {condition_name} "
                    "hilang atau bukan object."
                )

            if condition.get("sample_id") != sample_id:
                raise ValueError(
                    f"{sample_id}: sample_id pada {condition_name} "
                    "tidak cocok."
                )

            response = condition.get("response")

            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    f"{sample_id}: respons {condition_name} kosong."
                )

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Raw JSONL memiliki sample_id duplikat.")

    return sample_ids


def validate_classified_records(
    records: list[dict],
    expected_ids: list[str],
) -> Counter:
    """Validasi output klasifikasi dan hitung label per kondisi."""
    if len(records) != len(expected_ids):
        raise ValueError(
            f"Jumlah record output salah: {len(records)}; "
            f"diharapkan {len(expected_ids)}."
        )

    actual_ids = [record.get("sample_id") for record in records]

    if actual_ids != expected_ids:
        raise ValueError(
            "sample_id atau urutan output berbeda dari raw JSONL."
        )

    counts = Counter()

    for record in records:
        sample_id = record["sample_id"]

        for condition_name in CONDITIONS:
            condition = record.get(condition_name)

            if not isinstance(condition, dict):
                raise ValueError(
                    f"{sample_id}: kondisi {condition_name} "
                    "hilang setelah klasifikasi."
                )

            label = condition.get("classification")

            if label not in VALID_LABELS:
                raise ValueError(
                    f"{sample_id}/{condition_name}: "
                    f"label tidak valid: {label!r}"
                )

            counts[(condition_name, label)] += 1

    return counts


def main() -> None:
    print("=" * 70)
    print("JBB 100-SAMPLE CLASSIFICATION")
    print("=" * 70)

    print(f"\nInput raw : {INPUT_PATH}")
    print(f"Output    : {OUTPUT_PATH}")

    # 1. Baca dan validasi raw JSONL.
    raw_records = read_jsonl(INPUT_PATH)
    expected_ids = validate_raw_records(raw_records)

    print(f"\nValidated samples: {len(raw_records)}")
    print(f"Expected condition results: {len(raw_records) * 3}")

    # 2. Siapkan output klasifikasi yang bersih.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        print("Existing classified output removed for a clean rerun.")

    # 3. Jalankan classifier yang sudah ada.
    classifier = RuleBasedResponseClassifier()

    runner = ClassificationRunner(
        classifier=classifier,
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
    )

    print("\nRunning rule-based classification...")
    processed_count = runner.run()

    if processed_count != len(raw_records):
        raise RuntimeError(
            f"Runner memproses {processed_count} record; "
            f"diharapkan {len(raw_records)}."
        )

    # 4. Baca ulang dan validasi output yang tersimpan.
    classified_records = read_jsonl(OUTPUT_PATH)

    counts = validate_classified_records(
        classified_records,
        expected_ids,
    )

    # 5. Cetak distribusi label.
    print("\nClassification summary")
    print("-" * 70)
    print(
        f"{'Condition':<15}"
        f"{'Refusal':>10}"
        f"{'Compliance':>12}"
        f"{'Ambiguous':>12}"
    )

    for condition_name in CONDITIONS:
        refusal = counts[(condition_name, "refusal")]
        compliance = counts[(condition_name, "compliance")]
        ambiguous = counts[(condition_name, "ambiguous")]

        print(
            f"{condition_name:<15}"
            f"{refusal:>10}"
            f"{compliance:>12}"
            f"{ambiguous:>12}"
        )

    print("-" * 70)
    print(f"Classified samples: {processed_count}")
    print(f"Condition results: {processed_count * len(CONDITIONS)}")
    print(f"Classified output: {OUTPUT_PATH}")
    print("\nCLASSIFICATION PIPELINE PASSED")
    print(
        "Catatan: kelulusan pipeline hanya menunjukkan "
        "struktur dan label valid, bukan kebenaran semantik label."
    )


if __name__ == "__main__":
    main()
