from collections import Counter
import json
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
    / "smoke_test"
)

CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)

VALID_LABELS = {
    label.value for label in ClassificationLabel
}


def find_latest_raw_batch() -> Path:
    """Cari file raw batch terbaru, bukan file hasil klasifikasi."""
    candidates = [
        path
        for path in RESULTS_DIR.glob("jbb_real_batch*.jsonl")
        if not path.stem.endswith("_classified")
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Tidak ditemukan file raw batch di: {RESULTS_DIR}\n"
            "Jalankan manual batch smoke terlebih dahulu."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_jsonl(path: Path) -> list[dict]:
    """Baca JSONL dan laporkan nomor baris jika formatnya rusak."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid di {path}, "
                    f"baris {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record baris {line_number} harus berupa object JSON."
                )

            records.append(record)

    return records


def validate_raw_records(records: list[dict]) -> None:
    """Pastikan raw batch memiliki ID unik dan tiga kondisi."""
    if not records:
        raise ValueError("File raw batch kosong.")

    seen_ids = set()

    for index, record in enumerate(records, start=1):
        sample_id = record.get("sample_id")

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(
                f"Record ke-{index} tidak memiliki sample_id yang valid."
            )

        if sample_id in seen_ids:
            raise ValueError(
                f"sample_id duplikat di raw batch: {sample_id}"
            )
        seen_ids.add(sample_id)

        for condition_name in CONDITIONS:
            condition = record.get(condition_name)

            if not isinstance(condition, dict):
                raise ValueError(
                    f"{sample_id}: kondisi '{condition_name}' "
                    "hilang atau bukan object."
                )

            if condition.get("sample_id") != sample_id:
                raise ValueError(
                    f"{sample_id}: sample_id di '{condition_name}' "
                    "tidak cocok."
                )

            response = condition.get("response")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    f"{sample_id}: response '{condition_name}' kosong "
                    "atau bukan string."
                )


def validate_classified_records(
    raw_records: list[dict],
    classified_records: list[dict],
) -> Counter:
    """Periksa hasil klasifikasi dan hitung label per kondisi."""
    if len(raw_records) != len(classified_records):
        raise ValueError(
            "Jumlah record berubah setelah klasifikasi: "
            f"raw={len(raw_records)}, "
            f"classified={len(classified_records)}."
        )

    raw_ids = [record["sample_id"] for record in raw_records]
    classified_ids = [
        record.get("sample_id") for record in classified_records
    ]

    if raw_ids != classified_ids:
        raise ValueError(
            "Urutan atau sample_id output tidak cocok dengan input."
        )

    counts = Counter()

    for record in classified_records:
        sample_id = record["sample_id"]

        for condition_name in CONDITIONS:
            condition = record.get(condition_name)

            if not isinstance(condition, dict):
                raise ValueError(
                    f"{sample_id}: kondisi '{condition_name}' "
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
    print("AUTOMATIC CLASSIFICATION SMOKE TEST")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = find_latest_raw_batch()
    output_path = input_path.with_name(
        f"{input_path.stem}_classified.jsonl"
    )

    print(f"\nInput : {input_path}")
    print(f"Output: {output_path}")

    raw_records = read_jsonl(input_path)
    validate_raw_records(raw_records)

    print(f"\nRaw samples validated: {len(raw_records)}")
    print(
        f"Expected condition results: "
        f"{len(raw_records) * len(CONDITIONS)}"
    )

    # ClassificationRunner/Writer dapat mempertahankan output lama
    # jika writer menggunakan mode append. Hapus hanya file output
    # klasifikasi yang akan dibuat ulang; raw input tidak disentuh.
    if output_path.exists():
        output_path.unlink()
        print("Existing classified output removed for a clean rerun.")

    classifier = RuleBasedResponseClassifier()

    runner = ClassificationRunner(
        classifier=classifier,
        input_path=input_path,
        output_path=output_path,
    )

    processed_count = runner.run()

    if processed_count != len(raw_records):
        raise RuntimeError(
            f"Runner memproses {processed_count} record, "
            f"padahal input berisi {len(raw_records)}."
        )

    if not output_path.exists():
        raise RuntimeError("File output klasifikasi tidak ditemukan.")

    classified_records = read_jsonl(output_path)
    counts = validate_classified_records(
        raw_records,
        classified_records,
    )

    print("\nClassification summary")
    print("-" * 70)
    print(f"{'Condition':<15} {'Refusal':>10} {'Compliance':>12} {'Ambiguous':>12}")

    for condition_name in CONDITIONS:
        refusal = counts[(condition_name, "refusal")]
        compliance = counts[(condition_name, "compliance")]
        ambiguous = counts[(condition_name, "ambiguous")]

        print(
            f"{condition_name:<15} "
            f"{refusal:>10} "
            f"{compliance:>12} "
            f"{ambiguous:>12}"
        )

    print("-" * 70)
    print(f"Records classified: {processed_count}")
    print(f"Output: {output_path}")
    print("\nCLASSIFICATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()