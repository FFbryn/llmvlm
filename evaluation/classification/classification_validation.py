
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "jbb_100"
    / "jbb_100_classified.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "jbb_100"
    / "classification_validation.jsonl"
)

CONDITIONS = ("llm_text", "vlm_text", "vlm_image")
LABELS = ("REFUSAL", "COMPLIANCE", "AMBIGUOUS")

RANDOM_SEED = 42
MAX_PER_CONDITION_LABEL = 5


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL and report the exact line if parsing fails."""
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
                    f"Record pada baris {line_number} harus berupa objek JSON."
                )

            records.append(record)

    if not records:
        raise ValueError(f"File tidak memiliki record: {path}")

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def extract_label(condition_result: dict[str, Any]) -> str:
    """
    Support several common serialized classification formats.

    Examples:
      "classification": "REFUSAL"
      "classification": {"label": "REFUSAL"}
      "classification": {"label": {"value": "REFUSAL"}}
    """
    value = condition_result.get("classification")

    for _ in range(3):
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in LABELS:
                return normalized
            break

        if isinstance(value, dict):
            value = value.get("label", value.get("value"))
            continue

        break

    # Some pipelines may store the label directly.
    value = condition_result.get("label")
    if isinstance(value, str) and value.strip().upper() in LABELS:
        return value.strip().upper()

    raise ValueError(
        "Label otomatis tidak ditemukan atau formatnya tidak dikenal. "
        "Periksa field 'classification' pada record."
    )


def extract_response(condition_result: dict[str, Any]) -> str:
    response = condition_result.get("response")

    # Support a serialized response object as well as plain text.
    if isinstance(response, dict):
        response = response.get("text", response.get("content"))

    if not isinstance(response, str) or not response.strip():
        raise ValueError("Response kosong atau format response tidak dikenal.")

    return response.strip()


def flatten_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert each sample's three conditions into individual rows."""
    flattened = []
    seen_ids = set()

    for record in records:
        sample_id = record.get("sample_id")

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("Record tidak memiliki sample_id yang valid.")

        if sample_id in seen_ids:
            raise ValueError(f"sample_id duplikat: {sample_id}")

        seen_ids.add(sample_id)

        for condition_name in CONDITIONS:
            condition_result = record.get(condition_name)

            if not isinstance(condition_result, dict):
                raise ValueError(
                    f"{sample_id}: kondisi {condition_name} "
                    "tidak ditemukan atau bukan objek."
                )

            condition_sample_id = condition_result.get("sample_id")
            if (
                condition_sample_id is not None
                and condition_sample_id != sample_id
            ):
                raise ValueError(
                    f"sample_id tidak cocok pada {sample_id}, "
                    f"kondisi {condition_name}."
                )

            flattened.append(
                {
                    "sample_id": sample_id,
                    "condition": condition_name,
                    "response": extract_response(condition_result),
                    "auto_label": extract_label(condition_result),
                }
            )

    return flattened


def prepare_annotations(
    rows: list[dict[str, Any]],
    max_per_group: int = MAX_PER_CONDITION_LABEL,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """
    Stratified sampling by condition and automatic label.

    At most max_per_group rows are selected from each group.
    Groups with fewer rows contribute all available rows.
    """
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in rows:
        key = (row["condition"], row["auto_label"])
        groups.setdefault(key, []).append(row)

    selected = []

    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        selected.extend(group[:max_per_group])

    rng.shuffle(selected)

    annotations = []
    for index, row in enumerate(selected, start=1):
        annotations.append(
            {
                "annotation_id": f"ann_{index:04d}",
                **row,
                "human_label": "",
                "review_notes": "",
            }
        )

    return annotations


def validate_human_label(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("human_label harus berupa string.")

    label = value.strip().upper()

    if label not in LABELS:
        raise ValueError(
            f"human_label harus salah satu dari {LABELS}; "
            f"nilai yang ditemukan: {value!r}"
        )

    return label


def evaluate_annotations(
    annotations: list[dict[str, Any]],
) -> None:
    """Compare automatic labels with completed human annotations."""
    reviewed = []

    for row in annotations:
        human_value = row.get("human_label", "")

        if not isinstance(human_value, str) or not human_value.strip():
            continue

        auto_label = validate_human_label(row.get("auto_label"))
        human_label = validate_human_label(human_value)

        reviewed.append((auto_label, human_label))

    if not reviewed:
        raise ValueError(
            "Belum ada human_label yang diisi. "
            "Isi anotasi terlebih dahulu sebelum evaluasi."
        )

    correct = sum(
        auto_label == human_label
        for auto_label, human_label in reviewed
    )
    total = len(reviewed)
    accuracy = correct / total

    print("\n" + "=" * 60)
    print("CLASSIFICATION VALIDATION")
    print("=" * 60)
    print(f"Total baris anotasi : {len(annotations)}")
    print(f"Sudah ditinjau      : {total}")
    print(f"Belum ditinjau      : {len(annotations) - total}")
    print(f"Label sama          : {correct}")
    print(f"Accuracy            : {accuracy:.4f} ({accuracy:.1%})")

    print("\nConfusion matrix")
    print("Baris = label manusia; kolom = label otomatis")
    print(f"{'Human \\ Auto':<16}" + "".join(
        f"{label:>14}" for label in LABELS
    ))

    for human_label in LABELS:
        values = [
            sum(
                human == human_label and auto == auto_label
                for auto, human in reviewed
            )
            for auto_label in LABELS
        ]
        print(
            f"{human_label:<16}"
            + "".join(f"{value:>14}" for value in values)
        )

    print("\nPer-label precision, recall, dan F1")
    print("(Label otomatis dianggap sebagai prediksi)")

    for label in LABELS:
        tp = sum(
            auto == label and human == label
            for auto, human in reviewed
        )
        fp = sum(
            auto == label and human != label
            for auto, human in reviewed
        )
        fn = sum(
            auto != label and human == label
            for auto, human in reviewed
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        print(
            f"{label:<12} "
            f"precision={precision:.3f}  "
            f"recall={recall:.3f}  "
            f"F1={f1:.3f}"
        )

    print("\nDistribusi label manusia")
    for label, count in sorted(
        Counter(human for _, human in reviewed).items()
    ):
        print(f"{label:<12}: {count}")

    print(
        "\nCatatan: metrik ini hanya mewakili sampel yang sudah ditinjau. "
        "Jika sampel diambil secara stratified, accuracy sampel belum tentu "
        "mewakili accuracy keseluruhan dataset."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validasi manual hasil classification JSONL."
    )
    parser.add_argument(
        "--mode",
        choices=("prepare", "evaluate"),
        default="prepare",
        help="prepare membuat sampel anotasi; evaluate menghitung metrik.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path ke file JSONL hasil klasifikasi.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path file anotasi JSONL.",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=MAX_PER_CONDITION_LABEL,
        help="Maksimum sampel per kombinasi kondisi dan label otomatis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed agar pemilihan sampel dapat direproduksi.",
    )
    args = parser.parse_args()

    if args.mode == "prepare":
        if args.max_per_group < 1:
            raise ValueError("--max-per-group harus minimal 1.")

        if args.output.exists():
            raise FileExistsError(
                f"File anotasi sudah ada: {args.output}\n"
                "File tidak ditimpa untuk melindungi anotasi manual. "
                "Gunakan --output dengan nama file baru jika diperlukan."
            )

        raw_records = read_jsonl(args.input)
        rows = flatten_records(raw_records)
        annotations = prepare_annotations(
            rows,
            max_per_group=args.max_per_group,
            seed=args.seed,
        )
        write_jsonl(args.output, annotations)

        print("File anotasi berhasil dibuat.")
        print(f"Input             : {args.input}")
        print(f"Output            : {args.output}")
        print(f"Jumlah sampel asli: {len(raw_records)}")
        print(f"Jumlah respons    : {len(rows)}")
        print(f"Respons untuk review: {len(annotations)}")
        print("\nDistribusi sampel terpilih:")
        for key, count in sorted(
            Counter(
                (row["condition"], row["auto_label"])
                for row in annotations
            ).items()
        ):
            print(f"{key[0]:<12} {key[1]:<12}: {count}")

        print(
            "\nLangkah selanjutnya: isi field human_label pada file output "
            "dengan REFUSAL, COMPLIANCE, atau AMBIGUOUS."
        )

    else:
        annotations = read_jsonl(args.output)
        evaluate_annotations(annotations)


if __name__ == "__main__":
    main()
