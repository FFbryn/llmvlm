
import csv
import json
from collections import Counter
from pathlib import Path

from evaluation.classification.labels import ClassificationLabel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "benchmark_data" / "results" / "jbb_100"

INPUT_PATH = RESULTS_DIR / "jbb_100_classified.jsonl"
ANNOTATION_PATH = RESULTS_DIR / "ambiguous_annotations.csv"

PAIR_DETAIL_PATH = RESULTS_DIR / "three_condition_pair_details.csv"
PAIR_SUMMARY_PATH = RESULTS_DIR / "three_condition_pair_summary.csv"
PAIR_JSON_PATH = RESULTS_DIR / "three_condition_pair_summary.json"

CONDITIONS = ("llm_text", "vlm_text", "vlm_image")

PAIRS = (
    ("llm_text", "vlm_text"),
    ("vlm_text", "vlm_image"),
    ("llm_text", "vlm_image"),
)

VALID_LABELS = {label.value for label in ClassificationLabel}


def parse_label(value) -> str:
    if isinstance(value, dict):
        value = value.get("label", value.get("value"))

    if not isinstance(value, str):
        raise ValueError(f"Label tidak valid: {value!r}")

    normalized = value.strip().lower()

    if normalized not in VALID_LABELS:
        raise ValueError(f"Label tidak dikenal: {value!r}")

    return normalized


def read_jsonl() -> list[dict]:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"JSONL tidak ditemukan: {INPUT_PATH}")

    records = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada baris {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Baris {line_number} bukan objek JSON."
                )

            records.append(record)

    if len(records) != 100:
        raise ValueError(
            f"Diharapkan 100 sampel, ditemukan {len(records)}."
        )

    ids = [record.get("sample_id") for record in records]

    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("Ada sample_id yang tidak valid.")

    if len(ids) != len(set(ids)):
        raise ValueError("Ditemukan sample_id duplikat.")

    return records


def read_annotations() -> dict:
    if not ANNOTATION_PATH.is_file():
        raise FileNotFoundError(
            f"CSV anotasi tidak ditemukan: {ANNOTATION_PATH}"
        )

    annotations = {}

    with ANNOTATION_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "sample_id",
            "condition",
            "original_classification",
            "annotation_label",
        }

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "Kolom CSV anotasi tidak lengkap. "
                f"Kolom wajib: {sorted(required)}"
            )

        for row in reader:
            key = (row["sample_id"].strip(), row["condition"].strip())

            if key in annotations:
                raise ValueError(f"Anotasi duplikat: {key}")

            original = parse_label(row["original_classification"])

            if original != ClassificationLabel.AMBIGUOUS.value:
                raise ValueError(
                    f"{key}: file anotasi berisi label awal "
                    "yang bukan AMBIGUOUS."
                )

            label = (row.get("annotation_label") or "").strip().lower()

            if label:
                label = parse_label(label)

            annotations[key] = label

    return annotations


def resolve_labels(records: list[dict], annotations: dict) -> dict:
    resolved = {}
    expected_ambiguous_keys = set()

    for record in records:
        sample_id = record["sample_id"]

        for condition_name in CONDITIONS:
            condition = record.get(condition_name)

            if not isinstance(condition, dict):
                raise ValueError(
                    f"{sample_id}: kondisi {condition_name} tidak valid."
                )

            if condition.get("sample_id") != sample_id:
                raise ValueError(
                    f"{sample_id}: sample_id kondisi tidak cocok."
                )

            original = parse_label(condition.get("classification"))
            key = (sample_id, condition_name)

            if original == ClassificationLabel.AMBIGUOUS.value:
                expected_ambiguous_keys.add(key)
                annotated = annotations.get(key, "")

                resolved[key] = annotated or original
            else:
                resolved[key] = original

    unexpected = set(annotations) - expected_ambiguous_keys

    if unexpected:
        raise ValueError(
            "CSV anotasi berisi sampel/kondisi yang tidak cocok "
            f"dengan label AMBIGUOUS pada JSONL: {sorted(unexpected)[:5]}"
        )

    return resolved


def main() -> None:
    records = read_jsonl()
    annotations = read_annotations()
    resolved = resolve_labels(records, annotations)

    detail_rows = []

    for record in records:
        sample_id = record["sample_id"]

        for left, right in PAIRS:
            left_label = resolved[(sample_id, left)]
            right_label = resolved[(sample_id, right)]

            detail_rows.append(
                {
                    "sample_id": sample_id,
                    "pair": f"{left}_vs_{right}",
                    "condition_a": left,
                    "label_a": left_label,
                    "condition_b": right,
                    "label_b": right_label,
                    "same_label": left_label == right_label,
                    "pair_interpretation": (
                        "same_label"
                        if left_label == right_label
                        else "label_changed"
                    ),
                }
            )

    summary_rows = []

    for left, right in PAIRS:
        pair_name = f"{left}_vs_{right}"
        rows = [row for row in detail_rows if row["pair"] == pair_name]
        transitions = Counter(
            (row["label_a"], row["label_b"]) for row in rows
        )

        same_count = sum(row["same_label"] for row in rows)
        changed_count = len(rows) - same_count
        unresolved_count = sum(
            row["label_a"] == "ambiguous"
            or row["label_b"] == "ambiguous"
            for row in rows
        )

        summary_rows.append(
            {
                "pair": pair_name,
                "n": len(rows),
                "same_label_count": same_count,
                "same_label_rate": same_count / len(rows),
                "changed_label_count": changed_count,
                "changed_label_rate": changed_count / len(rows),
                "pairs_with_ambiguous_count": unresolved_count,
                "transitions_json": json.dumps(
                    {
                        f"{a}_to_{b}": count
                        for (a, b), count in sorted(transitions.items())
                    },
                    ensure_ascii=False,
                ),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with PAIR_DETAIL_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(detail_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    with PAIR_SUMMARY_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    PAIR_JSON_PATH.write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 65)
    print("THREE-CONDITION PAIR ANALYSIS")
    print("=" * 65)
    print(f"Samples: {len(records)}")
    print(f"Ambiguous annotation rows: {len(annotations)}")
    print(
        "Annotations filled:",
        sum(bool(value) for value in annotations.values()),
    )
    print(f"Pair detail: {PAIR_DETAIL_PATH}")
    print(f"Pair summary: {PAIR_SUMMARY_PATH}")
    print(f"Pair JSON: {PAIR_JSON_PATH}")

    for row in summary_rows:
        print(f"\n{row['pair']}")
        print(f"  N: {row['n']}")
        print(
            f"  Same label: {row['same_label_count']} "
            f"({row['same_label_rate']:.1%})"
        )
        print(
            f"  Changed label: {row['changed_label_count']} "
            f"({row['changed_label_rate']:.1%})"
        )
        print(
            f"  Pairs containing ambiguous: "
            f"{row['pairs_with_ambiguous_count']}"
        )
        print(f"  Transitions: {row['transitions_json']}")

    print("\nPAIR ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
