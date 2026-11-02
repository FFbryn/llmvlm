import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
)

INPUT_DIR = RESULTS_ROOT / "classified_100"

ANNOTATION_DIR = (
    RESULTS_ROOT
    / "ambiguous_annotations"
)

OUTPUT_ROOT = (
    RESULTS_ROOT
    / "transferability_analysis"
)


# ============================================================
# CONFIGURATION
# ============================================================

CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)

LABELS = (
    "REFUSAL",
    "COMPLIANCE",
    "AMBIGUOUS",
)

BINARY_LABELS = (
    "REFUSAL",
    "COMPLIANCE",
)

DATASETS = {
    "advbench": "advbench_100_classified.jsonl",
    "harmbench": "harmbench_100_classified.jsonl",
    "xstest": "xstest_100_classified.jsonl",
}

ANNOTATION_FILES = {
    name: f"{name}_100_ambiguous.csv"
    for name in DATASETS
}


# ============================================================
# JSONL LOADER
# ============================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL and validate its basic structure."""

    if not path.is_file():
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
                    f"JSON tidak valid di {path}, "
                    f"baris {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record baris {line_number} di {path} "
                    "bukan objek JSON."
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"File klasifikasi kosong: {path}"
        )

    return records


# ============================================================
# AUTOMATIC CLASSIFICATION LABEL
# ============================================================

def extract_label(
    condition_data: dict[str, Any],
) -> str:
    """
    Read a classification label stored as a string
    or inside a dictionary.
    """

    value = condition_data.get("classification")

    for _ in range(5):
        if isinstance(value, str):
            label = value.strip().upper()

            if label in LABELS:
                return label

            break

        if isinstance(value, dict):
            if "label" in value:
                value = value["label"]
            elif "value" in value:
                value = value["value"]
            else:
                break
        else:
            break

    value = condition_data.get("label")

    if isinstance(value, str):
        label = value.strip().upper()

        if label in LABELS:
            return label

    raise ValueError(
        "Label classification tidak ditemukan "
        "atau tidak valid."
    )


# ============================================================
# HUMAN ANNOTATION LABEL
# ============================================================

def normalize_human_label(
    value: str | None,
) -> str | None:
    """Normalize completed human labels."""

    if value is None or not value.strip():
        return None

    label = value.strip().upper()

    aliases = {
        "REFUSAL": "REFUSAL",
        "COMPLIANCE": "COMPLIANCE",
        "AMBIGUOUS": "AMBIGUOUS",
        "REFUSE": "REFUSAL",
        "COMPLY": "COMPLIANCE",
    }

    if label not in aliases:
        raise ValueError(
            f"annotation_label '{value}' tidak valid. "
            "Gunakan REFUSAL, COMPLIANCE, atau AMBIGUOUS."
        )

    return aliases[label]


# ============================================================
# LOAD ALL AUTOMATIC CLASSIFICATIONS
# ============================================================

def load_automatic_labels(
    path: Path,
) -> dict[str, dict[str, str]]:
    """
    Load labels for every sample and all three conditions.
    """

    labels = {}

    for record in read_jsonl(path):
        sample_id = record.get("sample_id")

        if (
            not isinstance(sample_id, str)
            or not sample_id.strip()
        ):
            raise ValueError(
                f"sample_id kosong/tidak valid pada {path}."
            )

        if sample_id in labels:
            raise ValueError(
                f"sample_id duplikat '{sample_id}' pada {path}."
            )

        labels[sample_id] = {}

        for condition in CONDITIONS:
            item = record.get(condition)

            if not isinstance(item, dict):
                raise ValueError(
                    f"Condition '{condition}' tidak ditemukan "
                    f"pada sample_id={sample_id}."
                )

            inner_id = item.get("sample_id")

            if (
                inner_id is not None
                and str(inner_id) != sample_id
            ):
                raise ValueError(
                    f"sample_id condition tidak cocok: "
                    f"{sample_id}/{condition}."
                )

            labels[sample_id][condition] = extract_label(item)

    return labels


# ============================================================
# LOAD HUMAN ANNOTATIONS FOR AMBIGUOUS RESPONSES
# ============================================================

def load_human_annotations(
    path: Path,
    automatic: dict[str, dict[str, str]],
):
    """
    Read the CSV produced by the ambiguous-export script.

    Only completed annotation_label values replace AMBIGUOUS.
    Blank annotation labels remain pending.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"File anotasi manusia tidak ditemukan:\n{path}"
        )

    human_labels = {}
    audit_rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "sample_id",
            "condition",
            "annotation_label",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Kolom wajib {sorted(missing)} "
                f"tidak ada pada {path.name}."
            )

        for line_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            condition = (row.get("condition") or "").strip()

            if (
                not sample_id
                or condition not in CONDITIONS
            ):
                raise ValueError(
                    f"sample_id/condition tidak valid "
                    f"di {path.name}, baris {line_number}."
                )

            if sample_id not in automatic:
                raise ValueError(
                    f"Sampel anotasi '{sample_id}' tidak ada "
                    f"di file klasifikasi {path.name}."
                )

            auto_label = automatic[sample_id][condition]

            if auto_label != "AMBIGUOUS":
                raise ValueError(
                    f"Anotasi {sample_id}/{condition} "
                    f"awalnya berlabel {auto_label}, "
                    "bukan AMBIGUOUS."
                )

            key = (sample_id, condition)

            if key in human_labels:
                raise ValueError(
                    f"Anotasi duplikat untuk "
                    f"{sample_id}/{condition} pada {path.name}."
                )

            human_label = normalize_human_label(
                row.get("annotation_label")
            )

            if human_label is not None:
                human_labels[key] = human_label

            audit_rows.append({
                "sample_id": sample_id,
                "condition": condition,
                "automatic_label": auto_label,
                "human_label": human_label or "",
                "annotation_status": (
                    "ANNOTATED"
                    if human_label is not None
                    else "PENDING"
                ),
                "annotation_notes": (
                    row.get("annotation_notes") or ""
                ).strip(),
                "annotator_id": (
                    row.get("annotator_id") or ""
                ).strip(),
            })

    return human_labels, audit_rows


# ============================================================
# STATISTICS HELPER
# ============================================================

def safe_rate(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


# ============================================================
# ANALYZE ONE DATASET
# ============================================================

def analyze_dataset(
    labels_by_sample: dict[str, dict[str, str]],
    dataset: str,
    input_path: Path,
    annotation_path: Path,
    annotation_rows: list[dict[str, str]],
):
    """
    Analyze all samples using the effective labels:
    automatic labels plus completed human annotations.
    """

    sample_ids = sorted(labels_by_sample)
    n_samples = len(sample_ids)

    if n_samples == 0:
        raise ValueError(
            f"Tidak ada sampel untuk dataset {dataset}."
        )

    # --------------------------------------------------------
    # 1. Label distribution per condition
    # --------------------------------------------------------

    per_condition = {}

    for condition in CONDITIONS:
        counts = Counter(
            labels_by_sample[sample_id][condition]
            for sample_id in sample_ids
        )

        non_ambiguous_n = (
            counts["REFUSAL"]
            + counts["COMPLIANCE"]
        )

        per_condition[condition] = {
            "counts": {
                label: counts[label]
                for label in LABELS
            },
            "rates_over_all_samples": {
                label: safe_rate(
                    counts[label],
                    n_samples,
                )
                for label in LABELS
            },
            "non_ambiguous_n": non_ambiguous_n,
            "compliance_rate_among_non_ambiguous": (
                safe_rate(
                    counts["COMPLIANCE"],
                    non_ambiguous_n,
                )
            ),
        }

    # --------------------------------------------------------
    # 2. Pairwise comparisons and transition matrices
    # --------------------------------------------------------

    pairwise_rows = []
    transition_rows = []

    for source in CONDITIONS:
        for target in CONDITIONS:
            if source == target:
                continue

            all_pairs = [
                (
                    labels_by_sample[sample_id][source],
                    labels_by_sample[sample_id][target],
                )
                for sample_id in sample_ids
            ]

            all_agree_n = sum(
                source_label == target_label
                for source_label, target_label in all_pairs
            )

            binary_pairs = [
                (source_label, target_label)
                for source_label, target_label in all_pairs
                if (
                    source_label in BINARY_LABELS
                    and target_label in BINARY_LABELS
                )
            ]

            binary_agree_n = sum(
                source_label == target_label
                for source_label, target_label in binary_pairs
            )

            source_compliant_known = [
                sample_id
                for sample_id in sample_ids
                if (
                    labels_by_sample[sample_id][source]
                    == "COMPLIANCE"
                    and labels_by_sample[sample_id][target]
                    in BINARY_LABELS
                )
            ]

            compliance_transfer_n = sum(
                labels_by_sample[sample_id][target]
                == "COMPLIANCE"
                for sample_id in source_compliant_known
            )

            source_compliance_target_ambiguous_n = sum(
                labels_by_sample[sample_id][source]
                == "COMPLIANCE"
                and labels_by_sample[sample_id][target]
                == "AMBIGUOUS"
                for sample_id in sample_ids
            )

            pairwise_rows.append({
                "dataset": dataset,
                "source_condition": source,
                "target_condition": target,
                "n_samples": n_samples,

                "all_label_agreement_n": all_agree_n,
                "all_label_agreement_rate": safe_rate(
                    all_agree_n,
                    len(all_pairs),
                ),

                "source_compliance_n": sum(
                    labels_by_sample[sample_id][source]
                    == "COMPLIANCE"
                    for sample_id in sample_ids
                ),

                "compliance_transfer_n": compliance_transfer_n,
                "compliance_transfer_denominator": len(
                    source_compliant_known
                ),
                "compliance_transfer_rate": safe_rate(
                    compliance_transfer_n,
                    len(source_compliant_known),
                ),

                "source_compliance_target_ambiguous_n": (
                    source_compliance_target_ambiguous_n
                ),

                "binary_comparable_n": len(binary_pairs),
                "binary_label_agreement_n": binary_agree_n,
                "binary_label_agreement_rate": safe_rate(
                    binary_agree_n,
                    len(binary_pairs),
                ),
            })

            # 3 x 3 transition matrix, including AMBIGUOUS.
            matrix = Counter(all_pairs)

            for source_label in LABELS:
                row_total = sum(
                    matrix[(source_label, target_label)]
                    for target_label in LABELS
                )

                for target_label in LABELS:
                    count = matrix[
                        (source_label, target_label)
                    ]

                    transition_rows.append({
                        "dataset": dataset,
                        "source_condition": source,
                        "target_condition": target,
                        "source_label": source_label,
                        "target_label": target_label,
                        "count": count,
                        "rate_within_source_label": safe_rate(
                            count,
                            row_total,
                        ),
                        "denominator_source_label": row_total,
                    })

    # --------------------------------------------------------
    # 3. Consistency across all three conditions
    # --------------------------------------------------------

    per_sample_rows = []

    for sample_id in sample_ids:
        row = {
            "dataset": dataset,
            "sample_id": sample_id,
        }

        for condition in CONDITIONS:
            row[condition] = (
                labels_by_sample[sample_id][condition]
            )

        all_three_binary = all(
            row[condition] in BINARY_LABELS
            for condition in CONDITIONS
        )

        row["all_three_binary"] = all_three_binary

        row["all_three_same_label"] = (
            all_three_binary
            and len({
                row[condition]
                for condition in CONDITIONS
            }) == 1
        )

        per_sample_rows.append(row)

    eligible = [
        row
        for row in per_sample_rows
        if row["all_three_binary"]
    ]

    same_n = sum(
        row["all_three_same_label"]
        for row in eligible
    )

    # --------------------------------------------------------
    # 4. Human annotation audit summary
    # --------------------------------------------------------

    annotation_status_counts = Counter(
        row["annotation_status"]
        for row in annotation_rows
    )

    human_changes = sum(
        1
        for row in annotation_rows
        if (
            row["human_label"]
            and row["human_label"] != row["automatic_label"]
        )
    )

    report = {
        "dataset": dataset,
        "classification_file": str(input_path),
        "human_annotation_file": str(annotation_path),
        "n_samples": n_samples,
        "conditions": list(CONDITIONS),
        "labels": list(LABELS),

        "human_annotation_summary": {
            "ambiguous_rows_in_annotation_csv": len(
                annotation_rows
            ),
            "completed_rows": annotation_status_counts[
                "ANNOTATED"
            ],
            "pending_rows": annotation_status_counts[
                "PENDING"
            ],
            "completed_human_labels_that_changed_ambiguous": (
                human_changes
            ),
            "policy": (
                "Filled annotation_label replaces automatic "
                "AMBIGUOUS for that sample and condition only. "
                "Blank annotation labels remain AMBIGUOUS."
            ),
        },

        "analysis_label_source": (
            "Automatic classifications for all samples, "
            "with completed human annotations replacing "
            "initial AMBIGUOUS labels."
        ),

        "ambiguity_policy": (
            "AMBIGUOUS remains a separate third label. "
            "All-label agreement and transition matrices "
            "include all samples. Binary agreement and "
            "compliance-transfer rates exclude ambiguous "
            "outcomes from their denominators."
        ),

        "per_condition": per_condition,
        "pairwise": pairwise_rows,

        "all_three_condition_consistency": {
            "n_samples_non_ambiguous_in_all_three": len(
                eligible
            ),
            "n_samples_same_binary_label_in_all_three": same_n,
            "same_label_rate": safe_rate(
                same_n,
                len(eligible),
            ),
        },

        "per_sample": per_sample_rows,
        "transitions": transition_rows,
    }

    return report


# ============================================================
# CSV WRITER
# ============================================================

def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write report rows to CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8-sig",
        )
        return

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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 76)
    print("TRANSFERABILITY ANALYSIS")
    print("AdvBench, HarmBench, XSTest")
    print("=" * 76)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Classification: {INPUT_DIR}")
    print(f"Human labels  : {ANNOTATION_DIR}")
    print(f"Output        : {OUTPUT_ROOT}")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    failures = []
    successful = 0
    combined_annotation_summary = []
    combined_pairwise = []

    for dataset, filename in DATASETS.items():
        input_path = INPUT_DIR / filename
        annotation_path = (
            ANNOTATION_DIR
            / ANNOTATION_FILES[dataset]
        )

        print("\n" + "=" * 76)
        print(f"DATASET: {dataset.upper()}")
        print("=" * 76)

        try:
            # 1. Load every automatic classification.
            automatic = load_automatic_labels(input_path)

            # 2. Load completed human labels for ambiguous rows.
            human_labels, audit_rows = load_human_annotations(
                annotation_path,
                automatic,
            )

            # 3. Combine labels.
            #    Human labels replace AMBIGUOUS only when filled.
            effective = {
                sample_id: dict(condition_labels)
                for sample_id, condition_labels
                in automatic.items()
            }

            for (sample_id, condition), human_label in (
                human_labels.items()
            ):
                effective[sample_id][condition] = human_label

            # 4. Analyze the complete dataset.
            report = analyze_dataset(
                labels_by_sample=effective,
                dataset=dataset,
                input_path=input_path,
                annotation_path=annotation_path,
                annotation_rows=audit_rows,
            )

            # 5. Save outputs for this dataset.
            dataset_output = OUTPUT_ROOT / dataset

            dataset_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            summary = {
                key: value
                for key, value in report.items()
                if key not in (
                    "per_sample",
                    "transitions",
                    "pairwise",
                )
            }

            summary_path = (
                dataset_output
                / "transferability_summary.json"
            )

            summary_path.write_text(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            write_csv(
                dataset_output / "transferability_pairwise.csv",
                report["pairwise"],
            )

            write_csv(
                dataset_output / "transferability_transitions.csv",
                report["transitions"],
            )

            write_csv(
                dataset_output / "transferability_per_sample.csv",
                report["per_sample"],
            )

            write_csv(
                dataset_output / "human_annotation_audit.csv",
                audit_rows,
            )

            # 6. Collect cross-dataset summaries.
            human_summary = report[
                "human_annotation_summary"
            ]

            combined_annotation_summary.append({
                "dataset": dataset,
                "n_samples": report["n_samples"],
                "annotation_rows": human_summary[
                    "ambiguous_rows_in_annotation_csv"
                ],
                "human_labels_completed": human_summary[
                    "completed_rows"
                ],
                "human_labels_pending": human_summary[
                    "pending_rows"
                ],
                "human_labels_changing_ambiguous": (
                    human_summary[
                        "completed_human_labels_that_changed_ambiguous"
                    ]
                ),
            })

            combined_pairwise.extend(
                report["pairwise"]
            )

            # 7. Print useful progress.
            print(f"Jumlah sampel: {report['n_samples']}")

            print(
                "Anotasi manusia: "
                f"{human_summary['completed_rows']} selesai, "
                f"{human_summary['pending_rows']} belum diisi"
            )

            print("\nDistribusi label setelah penggabungan:")

            for condition, details in (
                report["per_condition"].items()
            ):
                counts = details["counts"]

                print(
                    f"  {condition:<12} "
                    f"REFUSAL={counts['REFUSAL']:>3} "
                    f"COMPLIANCE={counts['COMPLIANCE']:>3} "
                    f"AMBIGUOUS={counts['AMBIGUOUS']:>3}"
                )

            print(f"\nOutput: {dataset_output}")

            successful += 1

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            failures.append((dataset, str(exc)))
            print(f"[GAGAL] {exc}")

    # --------------------------------------------------------
    # 8. Save combined reports
    # --------------------------------------------------------

    if combined_annotation_summary:
        write_csv(
            OUTPUT_ROOT / "all_datasets_annotation_summary.csv",
            combined_annotation_summary,
        )

        write_csv(
            OUTPUT_ROOT / "all_datasets_pairwise.csv",
            combined_pairwise,
        )

    print("\n" + "=" * 76)
    print("RINGKASAN AKHIR")
    print("=" * 76)

    print(
        f"Dataset berhasil: {successful}/{len(DATASETS)}"
    )

    for dataset, error in failures:
        print(f"{dataset}: GAGAL - {error}")

    if failures:
        raise SystemExit(1)

    print("\nSemua dataset berhasil dianalisis.")
    print(f"Folder hasil: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()