
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OVERLAP_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "dataset_overlap"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "selected_semantic_100"
)

# Candidate filenames are checked in order.
# JSONL is preferred; CSV and JSON are fallbacks.
INPUT_CANDIDATES = {
    "advbench": [
        "advbench_without_jbb_overlap.jsonl",
        "advbench_without_jbb_overlap.csv",
        "advbench_without_jbb_overlap.json",
        "advbench_selected_unique.jsonl",
        "advbench_selected_unique.csv",
        "advbench_selected_unique.json",
    ],
    "harmbench": [
        "harmbench_selected_unique.jsonl",
        "harmbench_selected_unique.csv",
        "harmbench_selected_unique.json",
    ],
    "xstest": [
        "xstest_selected_unique.jsonl",
        "xstest_selected_unique.csv",
        "xstest_selected_unique.json",
    ],
}

TEXT_FIELD_CANDIDATES = [
    "prompt",
    "goal",
    "text",
    "request",
    "instruction",
    "behavior",
    "harmful_behavior",
    "query",
    "sample",
]

METHOD_NAME = (
    "tfidf_word_char_cosine_farthest_first"
)


# ============================================================
# INPUT DISCOVERY
# ============================================================

def resolve_input_path(dataset_name):
    """
    Find the overlap-filtered dataset automatically.

    Priority:
    1. Known filenames in INPUT_CANDIDATES.
    2. Other files in the overlap directory with matching
       dataset name and a supported extension.

    JSONL is preferred over CSV when both are available.
    """
    candidates = INPUT_CANDIDATES[dataset_name]

    for filename in candidates:
        path = OVERLAP_DIR / filename
        if path.is_file():
            return path.resolve()

    # Fallback discovery for slightly different filenames.
    extension_priority = {
        ".jsonl": 0,
        ".csv": 1,
        ".json": 2,
    }

    discovered = []

    for path in OVERLAP_DIR.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in extension_priority:
            continue

        lower_name = path.name.casefold()

        if dataset_name not in lower_name:
            continue

        # Avoid accidentally selecting duplicate reports or raw files.
        excluded_tokens = [
            "duplicate",
            "overlap_report",
            "exact_duplicates",
            "normalized_duplicates",
            "audit",
        ]

        if any(token in lower_name for token in excluded_tokens):
            continue

        # Prefer candidate/selected/without-overlap files.
        # Do not fall back to arbitrary raw dataset files.
        preferred_tokens = [
            "selected_unique",
            "without_jbb_overlap",
            "without_overlap",
        ]

        if not any(token in lower_name for token in preferred_tokens):
            continue

        discovered.append(path)

    if discovered:
        discovered.sort(
            key=lambda path: (
                extension_priority[path.suffix.lower()],
                path.name.casefold(),
            )
        )
        return discovered[0].resolve()

    expected = "\n".join(
        f"  - {OVERLAP_DIR / name}"
        for name in candidates
    )

    raise FileNotFoundError(
        f"\nInput kandidat {dataset_name} tidak ditemukan.\n"
        f"Folder yang diperiksa: {OVERLAP_DIR}\n"
        f"Nama file yang dicari:\n{expected}\n"
        "Pastikan tahap seleksi overlap sudah dijalankan."
    )


# ============================================================
# FILE READING
# ============================================================

def read_input_file(path):
    """Read CSV, JSONL, or JSON into a list of dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(
            f"File input tidak ditemukan: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            rows = list(csv.DictReader(file))

    elif suffix == ".jsonl":
        rows = []

        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSONL tidak valid: {path}, "
                        f"baris {line_number}: {error}"
                    ) from error

                if not isinstance(item, dict):
                    raise ValueError(
                        f"Record pada baris {line_number} "
                        "harus berupa objek JSON."
                    )

                rows.append(item)

    elif suffix == ".json":
        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            rows = data

        elif isinstance(data, dict):
            rows = None

            for key in ("data", "samples", "records", "items"):
                if isinstance(data.get(key), list):
                    rows = data[key]
                    break

            if rows is None:
                raise ValueError(
                    f"Struktur JSON tidak dikenali: {path}. "
                    "Diharapkan list record atau key data, samples, "
                    "records, atau items."
                )
        else:
            raise ValueError(
                f"Struktur JSON tidak didukung: {path}"
            )

        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(
                f"Semua record JSON harus berupa objek: {path}"
            )

    else:
        raise ValueError(
            f"Format input tidak didukung: {path.suffix}"
        )

    if not rows:
        raise ValueError(
            f"File input kosong atau tidak memiliki record: {path}"
        )

    return rows


def sha256_file(path):
    """Calculate SHA-256 for input provenance."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# TEXT FIELD AND DUPLICATE HANDLING
# ============================================================

def normalize_text(value):
    """Normalize whitespace and case for exact duplicate detection."""
    value = str(value or "")
    return re.sub(r"\s+", " ", value).strip().casefold()


def find_text_field(rows, dataset_name):
    """Identify the prompt/text column."""
    fields = set()

    for row in rows:
        fields.update(row.keys())

    field_lookup = {
        str(field).strip().casefold(): field
        for field in fields
    }

    for candidate in TEXT_FIELD_CANDIDATES:
        field = field_lookup.get(candidate.casefold())

        if field is None:
            continue

        if any(
            normalize_text(row.get(field, ""))
            for row in rows
        ):
            return field

    raise ValueError(
        f"Tidak dapat menemukan kolom teks untuk {dataset_name}.\n"
        f"Kolom yang tersedia: {sorted(map(str, fields))}\n"
        "Tambahkan nama kolom yang benar ke TEXT_FIELD_CANDIDATES."
    )


def remove_duplicate_prompts(rows, text_field):
    """
    Remove empty prompts and exact duplicates after normalization.
    The first occurrence is preserved.
    """
    unique_rows = []
    seen_texts = set()
    removed_count = 0

    for row in rows:
        normalized = normalize_text(row.get(text_field, ""))

        if not normalized:
            removed_count += 1
            continue

        if normalized in seen_texts:
            removed_count += 1
            continue

        seen_texts.add(normalized)
        unique_rows.append(row)

    return unique_rows, removed_count


# ============================================================
# TF-IDF SIMILARITY
# ============================================================

def build_similarity_matrix(texts):
    """
    Combine two TF-IDF representations:
      - Word n-grams: 1-2
      - Character n-grams: 3-5

    Both representations receive equal weight.
    Similarity is measured using cosine similarity.
    """
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float64,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float64,
    )

    word_matrix = word_vectorizer.fit_transform(texts)
    char_matrix = char_vectorizer.fit_transform(texts)

    word_matrix = normalize(word_matrix, norm="l2")
    char_matrix = normalize(char_matrix, norm="l2")

    combined = hstack(
        [
            word_matrix * 0.5,
            char_matrix * 0.5,
        ],
        format="csr",
    )

    combined = normalize(combined, norm="l2")

    return cosine_similarity(combined)


# ============================================================
# DIVERSITY SELECTION
# ============================================================

def farthest_first_selection(similarity_matrix, target):
    """
    Deterministic farthest-first heuristic.

    First sample:
        The sample with the highest average similarity to all samples.

    Subsequent samples:
        Select the sample with the lowest maximum similarity
        to the samples already selected.

    This is a lexical diversity heuristic, not a guarantee of
    human-validated semantic diversity.
    """
    count = similarity_matrix.shape[0]

    if target < 1:
        raise ValueError("Target harus minimal 1.")

    if target > count:
        raise ValueError(
            f"Target {target} melebihi kandidat unik {count}."
        )

    average_similarity = similarity_matrix.mean(axis=1)
    first_index = int(np.argmax(average_similarity))

    selected_indices = [first_index]

    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[first_index] = True

    max_similarity = similarity_matrix[
        :, first_index
    ].copy()

    while len(selected_indices) < target:
        scores = max_similarity.copy()
        scores[selected_mask] = np.inf

        next_index = int(np.argmin(scores))

        selected_indices.append(next_index)
        selected_mask[next_index] = True

        max_similarity = np.maximum(
            max_similarity,
            similarity_matrix[:, next_index],
        )

    return selected_indices


# ============================================================
# OUTPUT WRITERS
# ============================================================

def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


def write_csv(path, rows):
    """Write records while retaining all encountered columns."""
    if not rows:
        raise ValueError(
            f"Tidak ada record untuk ditulis ke {path}"
        )

    fieldnames = []
    seen_fields = set()

    for row in rows:
        for key in row:
            if key not in seen_fields:
                fieldnames.append(key)
                seen_fields.add(key)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(dataset_name, input_path, output_dir, target):
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 70)
    print(f"Input: {input_path}")

    original_rows = read_input_file(input_path)
    input_hash = sha256_file(input_path)

    text_field = find_text_field(
        original_rows,
        dataset_name,
    )

    unique_rows, removed_count = remove_duplicate_prompts(
        original_rows,
        text_field,
    )

    print(f"Jumlah record input       : {len(original_rows)}")
    print(f"Kolom teks                : {text_field}")
    print(f"Duplikat/kosong dihapus   : {removed_count}")
    print(f"Kandidat unik              : {len(unique_rows)}")
    print(f"Target seleksi             : {target}")

    if len(unique_rows) < target:
        raise ValueError(
            f"{dataset_name}: kandidat unik hanya "
            f"{len(unique_rows)}, sedangkan target {target}. "
            "Tidak ada sampel pengganti yang dibuat."
        )

    texts = [
        str(row[text_field]).strip()
        for row in unique_rows
    ]

    similarity_matrix = build_similarity_matrix(texts)

    selected_indices = farthest_first_selection(
        similarity_matrix,
        target,
    )

    rank_by_index = {
        index: rank
        for rank, index in enumerate(
            selected_indices,
            start=1,
        )
    }

    selected_set = set(selected_indices)
    selected_rows = []

    # Preserve original columns and append selection metadata.
    for index in selected_indices:
        row = dict(unique_rows[index])

        row["selection_dataset"] = dataset_name
        row["selection_rank"] = rank_by_index[index]
        row["selection_method"] = METHOD_NAME
        row["selection_text_field"] = text_field

        selected_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)

    filename_stem = f"{dataset_name}_{target}"

    csv_path = output_dir / f"{filename_stem}.csv"
    jsonl_path = output_dir / f"{filename_stem}.jsonl"
    audit_path = (
        output_dir
        / f"{dataset_name}_similarity_audit.csv"
    )

    # Audit every unique candidate.
    audit_rows = []

    for index, row in enumerate(unique_rows):
        is_selected = index in selected_set

        if is_selected:
            rank = rank_by_index[index]

            if rank == 1:
                max_similarity_to_prior = ""
            else:
                previous_indices = selected_indices[:rank - 1]

                max_similarity_to_prior = float(
                    similarity_matrix[
                        index,
                        previous_indices,
                    ].max()
                )
        else:
            rank = ""
            max_similarity_to_prior = float(
                similarity_matrix[
                    index,
                    selected_indices,
                ].max()
            )

        audit_rows.append(
            {
                "dataset": dataset_name,
                "candidate_index": index,
                "text_field": text_field,
                "prompt_text": str(
                    row.get(text_field, "")
                ).strip(),
                "selected": is_selected,
                "selection_rank": rank,
                "max_similarity_to_selected": (
                    max_similarity_to_prior
                ),
            }
        )

    # Save the six primary outputs across three datasets.
    write_csv(csv_path, selected_rows)
    write_jsonl(jsonl_path, selected_rows)
    write_csv(audit_path, audit_rows)

    result = {
        "dataset": dataset_name,
        "input_path": str(input_path.resolve()),
        "input_sha256": input_hash,
        "input_records": len(original_rows),
        "text_field": text_field,
        "duplicate_or_empty_prompts_removed": removed_count,
        "unique_candidates": len(unique_rows),
        "target": target,
        "selected_records": len(selected_rows),
        "selection_method": METHOD_NAME,
        "output_csv": str(csv_path.resolve()),
        "output_jsonl": str(jsonl_path.resolve()),
        "audit_csv": str(audit_path.resolve()),
    }

    print(f"Jumlah terpilih            : {len(selected_rows)}")
    print(f"CSV                        : {csv_path}")
    print(f"JSONL                      : {jsonl_path}")
    print(f"Audit                      : {audit_path}")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automatically select diverse subsets from the "
            "overlap-filtered AdvBench, HarmBench, and XSTest."
        )
    )

    parser.add_argument(
        "--target",
        type=int,
        default=100,
        help="Target sampel per dataset (default: 100).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder output.",
    )

    args = parser.parse_args()

    if args.target < 1:
        parser.error("--target harus minimal 1.")

    print("=" * 70)
    print("SEMANTIC DIVERSITY DATASET SELECTION")
    print("=" * 70)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input folder : {OVERLAP_DIR}")
    print(f"Output folder: {args.output_dir.resolve()}")
    print(f"Target       : {args.target} per dataset")

    # Resolve all inputs before starting selection.
    # This avoids processing only some datasets when another
    # required input is missing.
    inputs = {}

    try:
        for dataset_name in (
            "advbench",
            "harmbench",
            "xstest",
        ):
            inputs[dataset_name] = resolve_input_path(
                dataset_name
            )
            print(
                f"Input {dataset_name:10s}: "
                f"{inputs[dataset_name]}"
            )

    except FileNotFoundError as error:
        parser.exit(
            status=2,
            message=f"\nERROR INPUT:\n{error}\n",
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # Process datasets independently.
    for dataset_name in (
        "advbench",
        "harmbench",
        "xstest",
    ):
        result = process_dataset(
            dataset_name=dataset_name,
            input_path=inputs[dataset_name],
            output_dir=args.output_dir,
            target=args.target,
        )
        results.append(result)

    manifest = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "input_directory": str(OVERLAP_DIR.resolve()),
        "output_directory": str(args.output_dir.resolve()),
        "target_per_dataset": args.target,
        "selection_method": METHOD_NAME,
        "method_details": {
            "word_tfidf_ngrams": [1, 2],
            "character_tfidf_ngrams": [3, 5],
            "word_weight": 0.5,
            "character_weight": 0.5,
            "similarity": "cosine",
            "selection": "deterministic farthest-first",
            "random_seed": None,
        },
        "notes": [
            "Inputs must be candidate datasets from the overlap stage.",
            "JSONL is preferred over CSV when both are available.",
            "Exact duplicates are removed after whitespace and "
            "case normalization.",
            "Original fields are preserved and selection metadata "
            "is appended.",
            "TF-IDF is a lexical diversity heuristic and does not "
            "guarantee human-validated semantic diversity.",
            "No model inference or new jailbreak prompts are generated.",
        ],
        "datasets": results,
    }

    manifest_path = (
        args.output_dir
        / "text_similarity_selection_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("SELEKSI SELESAI")
    print("=" * 70)

    total_selected = 0

    for result in results:
        print(
            f"{result['dataset']:10s}: "
            f"{result['selected_records']} sampel"
        )
        total_selected += result["selected_records"]

    print(f"\nTotal sampel terpilih: {total_selected}")
    print(f"Manifest: {manifest_path}")
    print(f"Output directory: {args.output_dir.resolve()}")

    print("\nEnam file utama:")
    for dataset_name in (
        "advbench",
        "harmbench",
        "xstest",
    ):
        stem = f"{dataset_name}_{args.target}"
        print(f"- {stem}.csv")
        print(f"- {stem}.jsonl")


if __name__ == "__main__":
    main()
