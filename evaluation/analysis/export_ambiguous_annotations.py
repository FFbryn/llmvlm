
import csv
import json
from pathlib import Path

from evaluation.classification.labels import ClassificationLabel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "benchmark_data" / "results" / "jbb_100"

INPUT_PATH = RESULTS_DIR / "jbb_100_classified.jsonl"
OUTPUT_PATH = RESULTS_DIR / "ambiguous_annotations.csv"

CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)


def parse_classification(value) -> ClassificationLabel:
    """Parse labels stored as strings or serialized dictionaries."""

    for _ in range(3):
        if isinstance(value, ClassificationLabel):
            return value

        if isinstance(value, str):
            try:
                return ClassificationLabel(value.strip().lower())
            except ValueError as exc:
                raise ValueError(
                    f"Label klasifikasi tidak dikenal: {value!r}"
                ) from exc

        if isinstance(value, dict):
            value = value.get("label", value.get("value"))
            continue

        break

    raise ValueError(
        f"Format classification tidak valid: {value!r}"
    )


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"File input tidak ditemukan: {INPUT_PATH}"
        )

    annotation_rows = []
    seen_sample_ids = set()

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada baris {line_number}."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record pada baris {line_number} bukan objek JSON."
                )

            sample_id = record.get("sample_id")

            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(
                    f"sample_id tidak valid pada baris {line_number}."
                )

            if sample_id in seen_sample_ids:
                raise ValueError(
                    f"sample_id duplikat: {sample_id}"
                )

            seen_sample_ids.add(sample_id)

            for condition_name in CONDITIONS:
                condition = record.get(condition_name)

                if not isinstance(condition, dict):
                    raise ValueError(
                        f"{sample_id}: kondisi {condition_name} "
                        "tidak ditemukan atau tidak valid."
                    )

                if condition.get("sample_id") != sample_id:
                    raise ValueError(
                        f"{sample_id}: sample_id kondisi "
                        f"{condition_name} tidak cocok."
                    )

                classification = parse_classification(
                    condition.get("classification")
                )

                if classification != ClassificationLabel.AMBIGUOUS:
                    continue

                response = condition.get("response")

                if not isinstance(response, str) or not response.strip():
                    raise ValueError(
                        f"{sample_id}: response {condition_name} kosong."
                    )

                model = condition.get("model", "")

                annotation_rows.append(
                    {
                        "sample_id": sample_id,
                        "condition": condition_name,
                        "model": model,
                        "original_classification": classification.value,
                        "response": response,
                        "annotation_label": "",
                        "annotation_notes": "",
                        "annotator_id": "",
                    }
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "condition",
        "model",
        "original_classification",
        "response",
        "annotation_label",
        "annotation_notes",
        "annotator_id",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotation_rows)

    print("=" * 65)
    print("AMBIGUOUS RESPONSE ANNOTATION EXPORT")
    print("=" * 65)
    print(f"Input samples       : {len(seen_sample_ids)}")
    print(f"Ambiguous responses : {len(annotation_rows)}")
    print(f"Output              : {OUTPUT_PATH}")
    print()
    print("Export selesai. File JSONL sumber tidak diubah.")


if __name__ == "__main__":
    main()
