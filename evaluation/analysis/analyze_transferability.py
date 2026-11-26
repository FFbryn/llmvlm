
import csv
import json
from collections import Counter
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "jbb_100"
)

INPUT_JSONL = RESULTS_DIR / "jbb_100_classified.jsonl"
ANNOTATIONS_CSV = RESULTS_DIR / "ambiguous_annotations.csv"

DETAILS_CSV = RESULTS_DIR / "transferability_details.csv"
METRICS_CSV = RESULTS_DIR / "transferability_metrics.csv"
VISUAL_ONLY_CSV = RESULTS_DIR / "visual_only_success_cases.csv"
SUMMARY_JSON = RESULTS_DIR / "transferability_summary.json"


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

VALID_LABELS = {
    "COMPLIANCE",
    "REFUSAL",
    "AMBIGUOUS",
}

CONDITIONS = (
    "llm_text",
    "vlm_text",
    "vlm_image",
)


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(value):
    """Normalize classification and human annotation labels."""

    if isinstance(value, dict):
        for key in (
            "label",
            "value",
            "classification",
            "name",
        ):
            if key in value:
                return normalize_label(value[key])

        return "AMBIGUOUS"

    if value is None:
        return "AMBIGUOUS"

    label = str(value).strip().upper()

    # Handle enum-style values.
    if "." in label:
        label = label.rsplit(".", 1)[-1]

    # Human annotation uses REFUSE.
    # Experiment classification uses REFUSAL.
    aliases = {
        "REFUSE": "REFUSAL",
    }

    label = aliases.get(label, label)

    if label not in VALID_LABELS:
        return "AMBIGUOUS"

    return label


# ============================================================
# SAMPLE ID
# ============================================================

def get_sample_id(record):

    for key in (
        "sample_id",
        "id",
        "benchmark_id",
    ):
        if record.get(key) is not None:
            return str(record[key])

    raise ValueError(
        "Sample record does not contain a recognized ID field: "
        f"{list(record.keys())}"
    )


# ============================================================
# GET EXPERIMENT CONDITIONS
# ============================================================

def get_conditions(record):
    """
    Read the three conditions directly from each JSONL record.

    Expected structure:
        {
            "sample_id": "...",
            "llm_text": {...},
            "vlm_text": {...},
            "vlm_image": {...}
        }
    """

    missing = [
        name
        for name in CONDITIONS
        if name not in record
    ]

    if missing:
        raise ValueError(
            f"Sample {get_sample_id(record)} "
            f"is missing conditions: {missing}"
        )

    conditions = {
        name: record[name]
        for name in CONDITIONS
    }

    return conditions


# ============================================================
# READ JSONL
# ============================================================

def read_jsonl(path):

    if not path.exists():
        raise FileNotFoundError(
            f"JSONL not found: {path}"
        )

    records = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found in {path}"
        )

    return records


# ============================================================
# READ HUMAN ANNOTATIONS
# ============================================================

def read_annotations(path):
    """
    Return:
        {(sample_id, condition): annotation_label}

    CSV uses semicolon delimiter.
    Blank annotations are not used as replacements.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Annotation CSV not found: {path}"
        )

    annotations = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter=";"
        )

        required = {
            "sample_id",
            "condition",
            "original_classification",
            "annotation_label",
        }

        actual_columns = set(
            reader.fieldnames or []
        )

        if not required.issubset(actual_columns):
            raise ValueError(
                "Annotation CSV must contain columns: "
                f"{sorted(required)}. "
                f"Found: {sorted(actual_columns)}"
            )

        for row_number, row in enumerate(
            reader,
            start=2
        ):

            sample_id = str(
                row.get("sample_id") or ""
            ).strip()

            condition = str(
                row.get("condition") or ""
            ).strip()

            raw_label = str(
                row.get("annotation_label") or ""
            ).strip()

            if not sample_id or not condition:
                raise ValueError(
                    f"Missing sample_id or condition "
                    f"at CSV row {row_number}"
                )

            if condition not in CONDITIONS:
                raise ValueError(
                    f"Unknown condition {condition!r} "
                    f"at CSV row {row_number}"
                )

            original = normalize_label(
                row.get("original_classification")
            )

            if original != "AMBIGUOUS":
                raise ValueError(
                    f"Annotation at row {row_number} "
                    "does not have original AMBIGUOUS classification."
                )

            # Skip blank annotations.
            if not raw_label:
                continue

            label = normalize_label(raw_label)

            # Reject unknown annotation labels.
            if label == "AMBIGUOUS" and (
                raw_label.strip().upper() != "AMBIGUOUS"
            ):
                raise ValueError(
                    f"Invalid annotation label "
                    f"{raw_label!r} at row {row_number}"
                )

            key = (
                sample_id,
                condition
            )

            if key in annotations:
                raise ValueError(
                    f"Duplicate annotation for {key}"
                )

            annotations[key] = label

    return annotations


# ============================================================
# EXTRACT ORIGINAL CLASSIFICATION
# ============================================================

def extract_label(condition_record):
    """Extract the original classification."""

    if not isinstance(condition_record, dict):
        return normalize_label(condition_record)

    for key in (
        "classification",
        "label",
        "outcome",
    ):
        if key in condition_record:
            return normalize_label(
                condition_record[key]
            )

    return "AMBIGUOUS"


# ============================================================
# RESOLVE CLASSIFICATION
# ============================================================

def resolve_label(
    condition_record,
    sample_id,
    condition,
    annotations
):

    original = extract_label(
        condition_record
    )

    annotation = annotations.get(
        (sample_id, condition)
    )

    # Human annotations replace only
    # originally ambiguous classifications.

    if (
        original == "AMBIGUOUS"
        and annotation is not None
    ):
        return (
            annotation,
            original,
            "human_annotation"
        )

    return (
        original,
        original,
        "existing_classification"
    )


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(
    path,
    rows,
    fieldnames
):

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# METRIC UTILITIES
# ============================================================

def percentage(
    numerator,
    denominator
):

    if denominator == 0:
        return None

    return round(
        100.0 * numerator / denominator,
        2
    )


def metric_row(
    name,
    numerator,
    denominator,
    unresolved=0,
    note=""
):

    return {
        "metric": name,
        "numerator": numerator,
        "denominator": denominator,
        "rate_percent": percentage(
            numerator,
            denominator
        ),
        "unresolved_count": unresolved,
        "note": note,
    }


# ============================================================
# COMPLIANCE TRANSFERABILITY
# ============================================================

def calculate_transfer_rate(
    rows,
    source,
    target,
    source_label
):
    """
    Strict rate:
    All eligible source samples form denominator.

    Resolved-only rate:
    Excludes ambiguous target outcomes.
    """

    eligible = [
        row
        for row in rows
        if row[source] == source_label
    ]

    unresolved = sum(
        row[target] == "AMBIGUOUS"
        for row in eligible
    )

    successes = sum(
        row[target] == "COMPLIANCE"
        for row in eligible
    )

    resolved = (
        len(eligible) - unresolved
    )

    return [

        metric_row(
            f"{source}_{source_label}_to_{target}_COMPLIANCE",

            successes,
            len(eligible),
            unresolved,

            "Strict rate; eligible source samples "
            "form the denominator."
        ),

        metric_row(
            f"{source}_{source_label}_to_{target}_COMPLIANCE_resolved_only",

            successes,
            resolved,
            unresolved,

            "Secondary rate among eligible samples "
            "with resolved target labels."
        ),

    ]


# ============================================================
# REFUSAL TO COMPLIANCE ANALYSIS
# ============================================================

def calculate_refusal_to_compliance(
    rows,
    source,
    target
):

    eligible = [
        row
        for row in rows
        if row[source] == "REFUSAL"
    ]

    unresolved = sum(
        row[target] == "AMBIGUOUS"
        for row in eligible
    )

    successes = sum(
        row[target] == "COMPLIANCE"
        for row in eligible
    )

    resolved = (
        len(eligible) - unresolved
    )

    return [

        metric_row(
            f"{source}_REFUSAL_to_{target}_COMPLIANCE",

            successes,
            len(eligible),
            unresolved,

            "Strict rate; denominator is all "
            "source REFUSAL samples."
        ),

        metric_row(
            f"{source}_REFUSAL_to_{target}_COMPLIANCE_resolved_only",

            successes,
            resolved,
            unresolved,

            "Secondary rate excluding ambiguous "
            "target labels."
        ),

    ]


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------------

    records = read_jsonl(
        INPUT_JSONL
    )

    annotations = read_annotations(
        ANNOTATIONS_CSV
    )

    detail_rows = []

    seen_ids = set()

    # --------------------------------------------------------
    # STEP 2: RESOLVE ALL CONDITIONS
    # --------------------------------------------------------

    for record in records:

        sample_id = get_sample_id(
            record
        )

        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate sample ID: {sample_id}"
            )

        seen_ids.add(sample_id)

        conditions = get_conditions(
            record
        )

        row = {
            "sample_id": sample_id
        }

        for condition in CONDITIONS:

            condition_record = conditions[
                condition
            ]

            if not isinstance(
                condition_record,
                dict
            ):
                raise ValueError(
                    f"{sample_id}: invalid condition "
                    f"record for {condition}"
                )

            if condition_record.get("sample_id") != sample_id:
                raise ValueError(
                    f"{sample_id}: sample_id mismatch "
                    f"in {condition}"
                )

            label, original, source = resolve_label(
                condition_record,
                sample_id,
                condition,
                annotations
            )

            row[condition] = label

            row[
                f"{condition}_original_label"
            ] = original

            row[
                f"{condition}_label_source"
            ] = source

        # ----------------------------------------------------
        # TRANSFERABILITY INDICATORS
        # ----------------------------------------------------

        row[
            "llm_compliance_to_vlm_text_compliance"
        ] = (
            row["llm_text"] == "COMPLIANCE"
            and row["vlm_text"] == "COMPLIANCE"
        )

        row[
            "llm_compliance_to_vlm_image_compliance"
        ] = (
            row["llm_text"] == "COMPLIANCE"
            and row["vlm_image"] == "COMPLIANCE"
        )

        row[
            "vlm_text_refusal_to_image_compliance"
        ] = (
            row["vlm_text"] == "REFUSAL"
            and row["vlm_image"] == "COMPLIANCE"
        )

        row[
            "llm_refusal_to_image_compliance"
        ] = (
            row["llm_text"] == "REFUSAL"
            and row["vlm_image"] == "COMPLIANCE"
        )

        row[
            "llm_refusal_to_image_refusal"
        ] = (
            row["llm_text"] == "REFUSAL"
            and row["vlm_image"] == "REFUSAL"
        )

        detail_rows.append(row)

    # --------------------------------------------------------
    # STEP 3: VALIDATE ANNOTATION REFERENCES
    # --------------------------------------------------------

    expected_ambiguous = set()

    for row in detail_rows:

        for condition in CONDITIONS:

            if (
                row[f"{condition}_original_label"]
                == "AMBIGUOUS"
            ):
                expected_ambiguous.add(
                    (row["sample_id"], condition)
                )

    unexpected_annotations = (
        set(annotations) - expected_ambiguous
    )

    if unexpected_annotations:
        raise ValueError(
            "Annotations do not match originally "
            "AMBIGUOUS JSONL conditions: "
            f"{sorted(unexpected_annotations)[:5]}"
        )

    # --------------------------------------------------------
    # STEP 4: LABEL DISTRIBUTION
    # --------------------------------------------------------

    label_counts = {

        condition: dict(
            Counter(
                row[condition]
                for row in detail_rows
            )
        )

        for condition in CONDITIONS

    }

    # --------------------------------------------------------
    # STEP 5: CALCULATE METRICS
    # --------------------------------------------------------

    metrics = []

    # Primary transferability metrics.

    for source, target in (
        ("llm_text", "vlm_text"),
        ("llm_text", "vlm_image"),
    ):

        metrics.extend(
            calculate_transfer_rate(
                detail_rows,
                source,
                target,
                "COMPLIANCE"
            )
        )

    # Refusal-to-compliance analysis.

    metrics.extend(
        calculate_refusal_to_compliance(
            detail_rows,
            "llm_text",
            "vlm_image"
        )
    )

    metrics.extend(
        calculate_refusal_to_compliance(
            detail_rows,
            "vlm_text",
            "vlm_image"
        )
    )

    # --------------------------------------------------------
    # STEP 6: REFUSAL CONSISTENCY
    # --------------------------------------------------------

    source_refusals = [

        row
        for row in detail_rows
        if row["llm_text"] == "REFUSAL"

    ]

    image_refusals = sum(
        row["vlm_image"] == "REFUSAL"
        for row in source_refusals
    )

    image_unresolved = sum(
        row["vlm_image"] == "AMBIGUOUS"
        for row in source_refusals
    )

    metrics.append(
        metric_row(
            "llm_text_REFUSAL_to_vlm_image_REFUSAL",

            image_refusals,
            len(source_refusals),
            image_unresolved,

            "Refusal consistency; not proof of "
            "overall guardrail effectiveness."
        )
    )

    # --------------------------------------------------------
    # STEP 7: VISUAL-ONLY SUCCESS CASES
    # --------------------------------------------------------

    visual_only = [

        row
        for row in detail_rows

        if (
            row["llm_text"] == "REFUSAL"
            and row["vlm_image"] == "COMPLIANCE"
        )

    ]

    # --------------------------------------------------------
    # STEP 8: SAVE RESULTS
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    detail_fields = [

        "sample_id",

        "llm_text",
        "vlm_text",
        "vlm_image",

        "llm_text_original_label",
        "vlm_text_original_label",
        "vlm_image_original_label",

        "llm_text_label_source",
        "vlm_text_label_source",
        "vlm_image_label_source",

        "llm_compliance_to_vlm_text_compliance",
        "llm_compliance_to_vlm_image_compliance",

        "vlm_text_refusal_to_image_compliance",

        "llm_refusal_to_image_compliance",
        "llm_refusal_to_image_refusal",

    ]

    write_csv(
        DETAILS_CSV,
        detail_rows,
        detail_fields
    )

    write_csv(
        METRICS_CSV,
        metrics,
        [
            "metric",
            "numerator",
            "denominator",
            "rate_percent",
            "unresolved_count",
            "note",
        ]
    )

    write_csv(
        VISUAL_ONLY_CSV,
        visual_only,
        detail_fields
    )

    # --------------------------------------------------------
    # STEP 9: SUMMARY JSON
    # --------------------------------------------------------

    summary = {

        "dataset": "JBB-100",

        "sample_count": len(detail_rows),

        "annotation_count": len(annotations),

        "label_counts_by_condition": label_counts,

        "visual_only_success_count": len(visual_only),

        "metrics": metrics,

        "interpretation_warning": (
            "Rates describe observed classifications under this "
            "experimental setup. They are not causal proof that visual "
            "modality alone caused a behavior change."
        ),

    }

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # STEP 10: PRINT RESULTS
    # --------------------------------------------------------

    print("=" * 72)
    print("TRANSFERABILITY ANALYSIS")
    print("=" * 72)

    print(
        f"Samples analyzed: {len(detail_rows)}"
    )

    print(
        f"Annotations loaded: {len(annotations)}"
    )

    print("\nLabel counts by condition:")

    for condition, counts in label_counts.items():

        print(
            f"  {condition}: {counts}"
        )

    print("\nMetrics:")

    for metric in metrics:

        rate = metric["rate_percent"]

        rate_text = (
            "N/A"
            if rate is None
            else f"{rate:.2f}%"
        )

        print(
            f"  {metric['metric']}: "
            f"{metric['numerator']}/"
            f"{metric['denominator']} "
            f"({rate_text}); "
            f"unresolved={metric['unresolved_count']}"
        )

    print(
        "\nVisual-only success cases:",
        len(visual_only)
    )

    print("\nOutput files:")

    print(f"  {DETAILS_CSV}")
    print(f"  {METRICS_CSV}")
    print(f"  {VISUAL_ONLY_CSV}")
    print(f"  {SUMMARY_JSON}")

    print("\n" + "=" * 72)
    print("TRANSFERABILITY ANALYSIS COMPLETED")
    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
