
import csv
import json
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIRS = {
    "jbb": (
        PROJECT_ROOT
        / "benchmark_data"
        / "results"
        / "classified_100"
    ),
    "advbench": (
        PROJECT_ROOT
        / "benchmark_data"
        / "results"
        / "classified_100"
    ),
    "harmbench": (
        PROJECT_ROOT
        / "benchmark_data"
        / "results"
        / "classified_100"
    ),
    "xstest": (
        PROJECT_ROOT
        / "benchmark_data"
        / "results"
        / "classified_100"
    ),
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "ambiguous_annotations"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_SAMPLE_COUNT = 100

CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)

DATASETS = {
    "jbb": "jbb_100_vicuna_llava_classified.jsonl",
    "advbench": "advbench_100_vicuna_llava_classified.jsonl",
    "harmbench": "harmbench_100_vicuna_llava_classified.jsonl",
    "xstest": "xstest_100_vicuna_llava_classified.jsonl",
}

AMBIGUOUS_LABEL = "ambiguous"

CSV_FIELDS = [
    "sample_id",
    "condition",
    "model",
    "original_classification",
    "response",
    "annotation_label",
    "annotation_notes",
    "annotator_id",
]


# ============================================================
# JSONL LOADER
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL records and validate their basic structure."""

    if not path.exists():
        raise FileNotFoundError(
            f"File klasifikasi tidak ditemukan:\n{path}"
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
                    f"JSON tidak valid pada {path.name}, "
                    f"baris {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record pada baris {line_number} "
                    f"di {path.name} bukan JSON object."
                )

            records.append(record)

    return records


# ============================================================
# CLASSIFICATION LABEL PARSER
# ============================================================

def get_label(condition_data: dict) -> str | None:
    """
    Read classification labels represented as:
    - a string, e.g. "ambiguous"
    - a dictionary containing "label" or "value"
    """

    classification = condition_data.get("classification")

    if isinstance(classification, str):
        return classification.strip().lower()

    if isinstance(classification, dict):
        label = classification.get(
            "label",
            classification.get("value"),
        )

        if isinstance(label, str):
            return label.strip().lower()

    return None


# ============================================================
# MODEL NAME HELPER
# ============================================================

def get_model_name(record: dict, condition: str) -> str:
    """Extract model name when available; otherwise leave blank."""

    condition_data = record.get(condition, {})

    if not isinstance(condition_data, dict):
        return ""

    model = condition_data.get("model")

    if isinstance(model, str):
        return model

    # Support an alternative schema if model metadata is stored
    # under a top-level dictionary.
    models = record.get("models")

    if isinstance(models, dict):
        model = models.get(condition)

        if isinstance(model, str):
            return model

        if isinstance(model, dict):
            name = model.get("name")

            if isinstance(name, str):
                return name

    return ""


# ============================================================
# VALIDATE CLASSIFIED RECORDS
# ============================================================

def validate_records(
    records: list[dict],
    input_path: Path,
) -> None:
    """Validate sample IDs and the three expected conditions."""

    if not records:
        raise ValueError(
            f"Tidak ada record pada {input_path.name}."
        )

    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Jumlah record tidak sesuai pada {input_path.name}. "
            f"Diharapkan {EXPECTED_SAMPLE_COUNT}, "
            f"ditemukan {len(records)}."
        )

    sample_ids = []

    for index, record in enumerate(records, start=1):
        sample_id = record.get("sample_id")

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(
                f"sample_id kosong/tidak valid pada record "
                f"{index} di {input_path.name}."
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

            if not isinstance(response, str):
                raise ValueError(
                    f"Response bukan string pada sample_id="
                    f"{sample_id}, condition={condition}."
                )

            if get_label(condition_data) is None:
                raise ValueError(
                    f"Label classification kosong/tidak valid "
                    f"pada sample_id={sample_id}, "
                    f"condition={condition}."
                )

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(
            f"sample_id duplikat ditemukan pada {input_path.name}."
        )


# ============================================================
# EXPORT AMBIGUOUS FOR ONE DATASET
# ============================================================

def export_dataset(
    dataset_name: str,
    input_filename: str,
) -> int:
    """Export ambiguous conditions to a dataset-specific CSV."""

    input_path = (
        INPUT_DIRS[dataset_name]
        / input_filename
    )

    output_path = (
        OUTPUT_DIR
        / f"{dataset_name}_100_vicuna_llava_ambiguous.csv"
    )

    print()
    print("=" * 72)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 72)

    records = load_jsonl(input_path)

    validate_records(
        records,
        input_path,
    )

    rows = []

    for record in records:
        sample_id = record["sample_id"]

        for condition in CONDITIONS:
            condition_data = record[condition]
            label = get_label(condition_data)

            if label != AMBIGUOUS_LABEL:
                continue

            rows.append({
                "sample_id": sample_id,
                "condition": condition,
                "model": get_model_name(record, condition),
                "original_classification": label,
                "response": condition_data["response"],
                "annotation_label": "",
                "annotation_notes": "",
                "annotator_id": "",
            })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write a fresh CSV on each run, not append to an old file.
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Jumlah record sampel       : {len(records)}")
    print(f"Jumlah baris ambiguous     : {len(rows)}")
    print(f"Output                     : {output_path}")
    print("Status                     : BERHASIL")

    return len(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 72)
    print("EXPORT AMBIGUOUS UNTUK HUMAN ANNOTATION")
    print("MODEL PAIR: VICUNA - LLAVA")
    print("=" * 72)

    print(f"Input directory : {PROJECT_ROOT / 'benchmark_data/results/classified_100'}")
    print(f"Output directory: {OUTPUT_DIR}")

    results = {}
    failures = {}

    for dataset_name, input_filename in DATASETS.items():
        try:
            results[dataset_name] = export_dataset(
                dataset_name,
                input_filename,
            )

        except Exception as exc:
            failures[dataset_name] = str(exc)
            print(
                f"\nGAGAL memproses {dataset_name}: {exc}"
            )

    print()
    print("=" * 72)
    print("RINGKASAN AKHIR")
    print("=" * 72)

    for dataset_name, count in results.items():
        print(
            f"{dataset_name:<12}: BERHASIL, "
            f"{count} baris ambiguous"
        )

    for dataset_name, error in failures.items():
        print(
            f"{dataset_name:<12}: GAGAL - {error}"
        )

    print(
        f"\nDataset berhasil: {len(results)}/{len(DATASETS)}"
    )

    if failures:
        print(
            "Periksa error di atas. Dataset yang gagal "
            "belum berhasil diekspor."
        )
    else:
        total_ambiguous = sum(results.values())

        print(
            f"Total baris ambiguous: {total_ambiguous}"
        )
        print(
            "Semua dataset berhasil diproses."
        )


if __name__ == "__main__":
    main()
