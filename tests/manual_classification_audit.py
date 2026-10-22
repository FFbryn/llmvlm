import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "benchmark_data" / "results"

CONDITIONS = ("llm_text", "vlm_text", "vlm_image")
VALID_LABELS = {"refusal", "compliance", "ambiguous"}


def find_latest_classified_file() -> Path:
    candidates = [
        path
        for path in RESULTS_DIR.rglob("*classified*.jsonl")
        if path.is_file()
        and "review" not in path.stem.lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Tidak ditemukan file classified JSONL di {RESULTS_DIR}"
        )

    # Pilih file yang terakhir diubah.
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_records(path: Path) -> list[dict]:
    records = []
    seen_ids = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada baris {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record baris {line_number} bukan object JSON."
                )

            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(
                    f"sample_id tidak valid pada baris {line_number}."
                )

            if sample_id in seen_ids:
                raise ValueError(
                    f"sample_id duplikat ditemukan: {sample_id}"
                )

            seen_ids.add(sample_id)
            records.append(record)

    return records


def audit_records(records: list[dict]):
    counts = {
        condition: Counter()
        for condition in CONDITIONS
    }
    review_items = []

    for record in records:
        sample_id = record["sample_id"]

        for condition_name in CONDITIONS:
            condition = record.get(condition_name)

            if not isinstance(condition, dict):
                review_items.append({
                    "sample_id": sample_id,
                    "condition": condition_name,
                    "reason": "missing_or_invalid_condition",
                    "response": None,
                    "classification": None,
                })
                continue

            label = condition.get("classification")
            response = condition.get("response")

            if label not in VALID_LABELS:
                review_items.append({
                    "sample_id": sample_id,
                    "condition": condition_name,
                    "reason": "missing_or_unknown_label",
                    "response": response,
                    "classification": label,
                })
                continue

            counts[condition_name][label] += 1

            # Ambiguous harus ditinjau; compliance juga perlu
            # diperiksa secara manual sebelum analisis final.
            if label == "ambiguous":
                reason = "ambiguous_response"
            elif label == "compliance":
                reason = "verify_compliance"
            else:
                reason = None

            if reason:
                review_items.append({
                    "sample_id": sample_id,
                    "condition": condition_name,
                    "reason": reason,
                    "response": response,
                    "classification": label,
                })

    return counts, review_items


def main():
    print("=" * 70)
    print("CLASSIFICATION AUDIT")
    print("=" * 70)

    input_path = find_latest_classified_file()
    records = load_records(input_path)

    print(f"Input file : {input_path}")
    print(f"Sample count: {len(records)}")

    counts, review_items = audit_records(records)

    print("\nLabel distribution")
    print("-" * 70)
    print(f"{'Condition':<15}{'Refusal':>12}{'Compliance':>14}{'Ambiguous':>14}")

    for condition in CONDITIONS:
        row = counts[condition]
        print(
            f"{condition:<15}"
            f"{row['refusal']:>12}"
            f"{row['compliance']:>14}"
            f"{row['ambiguous']:>14}"
        )

    review_path = input_path.with_name(
        input_path.stem + "_review.jsonl"
    )

    with review_path.open("w", encoding="utf-8") as file:
        for item in review_items:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    print(f"\nReview items: {len(review_items)}")
    print(f"Review file : {review_path}")
    print("\nCLASSIFICATION AUDIT COMPLETED")


if __name__ == "__main__":
    main()