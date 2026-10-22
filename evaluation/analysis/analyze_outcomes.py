
import csv
import json
from collections import Counter
from pathlib import Path

from evaluation.classification.labels import ClassificationLabel
from evaluation.outcome import SafetyOutcome


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "jbb_100"
)

INPUT_PATH = RESULTS_DIR / "jbb_100_classified.jsonl"
DETAIL_PATH = RESULTS_DIR / "jbb_100_outcomes.csv"
SUMMARY_PATH = RESULTS_DIR / "outcome_summary.csv"
JSON_SUMMARY_PATH = RESULTS_DIR / "outcome_summary.json"

EXPECTED_SAMPLE_COUNT = 100

CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"File input tidak ditemukan: {path}"
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada baris "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Baris {line_number} bukan objek JSON."
                )

            records.append(record)

    return records


def parse_classification(value) -> ClassificationLabel:
    """Parse a classification label from serialized JSON data."""

    for _ in range(3):
        if isinstance(value, ClassificationLabel):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()

            try:
                return ClassificationLabel(normalized)
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


def flatten_outcomes(records: list[dict]) -> list[dict]:
    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Diharapkan {EXPECTED_SAMPLE_COUNT} sampel, "
            f"ditemukan {len(records)}."
        )

    sample_ids = [
        record.get("sample_id")
        for record in records
    ]

    if any(
        not isinstance(sample_id, str) or not sample_id.strip()
        for sample_id in sample_ids
    ):
        raise ValueError(
            "Semua record harus memiliki sample_id valid."
        )

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            "Input memiliki sample_id duplikat."
        )

    rows = []

    for record in records:
        sample_id = record["sample_id"]

        for condition_name in CONDITIONS:
            condition = record.get(condition_name)

            if not isinstance(condition, dict):
                raise ValueError(
                    f"{sample_id}: kondisi {condition_name} "
                    "tidak valid."
                )

            if condition.get("sample_id") != sample_id:
                raise ValueError(
                    f"{sample_id}: sample_id pada "
                    f"{condition_name} hilang atau tidak cocok."
                )

            response = condition.get("response")

            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    f"{sample_id}: respons {condition_name} kosong."
                )

            classification = parse_classification(
                condition.get("classification")
            )

            safety_outcome = SafetyOutcome.from_classification(
                classification
            )

            rows.append(
                {
                    "sample_id": sample_id,
                    "condition": condition_name,
                    "classification": classification.value,
                    "outcome": safety_outcome.value,
                }
            )

    expected_rows = EXPECTED_SAMPLE_COUNT * len(CONDITIONS)

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Diharapkan {expected_rows} hasil, "
            f"ditemukan {len(rows)}."
        )

    return rows


def build_summary(rows: list[dict]) -> list[dict]:
    summary = []

    for condition_name in CONDITIONS:
        condition_rows = [
            row for row in rows
            if row["condition"] == condition_name
        ]

        total = len(condition_rows)

        if total == 0:
            raise ValueError(
                f"Tidak ada hasil untuk kondisi {condition_name}."
            )

        label_counts = Counter(
            row["classification"]
            for row in condition_rows
        )

        outcome_counts = Counter(
            row["outcome"]
            for row in condition_rows
        )

        summary.append(
            {
                "condition": condition_name,
                "n": total,
                "refusal_count": label_counts[
                    ClassificationLabel.REFUSAL.value
                ],
                "refusal_rate": (
                    label_counts[ClassificationLabel.REFUSAL.value]
                    / total
                ),
                "compliance_count": label_counts[
                    ClassificationLabel.COMPLIANCE.value
                ],
                "compliance_rate": (
                    label_counts[ClassificationLabel.COMPLIANCE.value]
                    / total
                ),
                "ambiguous_count": label_counts[
                    ClassificationLabel.AMBIGUOUS.value
                ],
                "ambiguous_rate": (
                    label_counts[ClassificationLabel.AMBIGUOUS.value]
                    / total
                ),
                "safety_failure_count": outcome_counts[
                    SafetyOutcome.FAILURE.value
                ],
                "safety_success_count": outcome_counts[
                    SafetyOutcome.SUCCESS.value
                ],
                "safety_uncertain_count": outcome_counts[
                    SafetyOutcome.UNCERTAIN.value
                ],
            }
        )

    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(
            f"Tidak ada data untuk ditulis ke {path}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 70)
    print("JBB-100 SAFETY OUTCOME ANALYSIS")
    print("=" * 70)

    records = read_jsonl(INPUT_PATH)
    rows = flatten_outcomes(records)
    summary = build_summary(rows)

    write_csv(DETAIL_PATH, rows)
    write_csv(SUMMARY_PATH, summary)

    JSON_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Input samples : {len(records)}")
    print(f"Outcome rows  : {len(rows)}")
    print(f"Detail CSV    : {DETAIL_PATH}")
    print(f"Summary CSV   : {SUMMARY_PATH}")
    print(f"Summary JSON  : {JSON_SUMMARY_PATH}")

    print("\nSafety outcome summary")
    print("-" * 70)

    for item in summary:
        print(f"\nCondition: {item['condition']}")
        print(f"  N                 : {item['n']}")
        print(
            f"  Refusal           : {item['refusal_count']} "
            f"({item['refusal_rate']:.1%})"
        )
        print(
            f"  Compliance        : {item['compliance_count']} "
            f"({item['compliance_rate']:.1%})"
        )
        print(
            f"  Ambiguous         : {item['ambiguous_count']} "
            f"({item['ambiguous_rate']:.1%})"
        )
        print(
            f"  Safety success    : {item['safety_success_count']}"
        )
        print(
            f"  Safety failure    : {item['safety_failure_count']}"
        )
        print(
            f"  Safety uncertain  : {item['safety_uncertain_count']}"
        )

    if sum(item["n"] for item in summary) != len(rows):
        raise RuntimeError(
            "Total pada ringkasan tidak cocok."
        )

    print("\nOUTCOME ANALYSIS PASSED")
    print(
        "Catatan: hasil ini mengikuti label classifier yang tersedia. "
        "Validitas perhitungan tidak membuktikan kebenaran semantik label."
    )


if __name__ == "__main__":
    main()
