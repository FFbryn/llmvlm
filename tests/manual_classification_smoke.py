import json
from pathlib import Path

from evaluation.classification import (
    ClassificationRunner,
    RuleBasedResponseClassifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "smoke_test"
)


def find_latest_input() -> Path:
    """
    Mencari file JSONL hasil batch JBB terbaru.

    File hasil klasifikasi tidak boleh dipilih sebagai input.
    """
    candidates = [
        path
        for path in RESULTS_DIR.glob("jbb_real_batch*.jsonl")
        if not path.stem.endswith("_classified")
        and path.is_file()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Tidak ditemukan file batch JBB di {RESULTS_DIR}. "
            "Jalankan batch smoke test terlebih dahulu."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def validate_input(path: Path) -> tuple[int, set[str]]:
    """
    Memvalidasi JSONL mentah sebelum klasifikasi.
    Memastikan setiap sampel memiliki tiga kondisi.
    """
    required_conditions = (
        "llm_text",
        "vlm_text",
        "vlm_image",
    )

    sample_ids = set()
    record_count = 0

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
                    f"Record pada baris {line_number} bukan object JSON."
                )

            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(
                    f"sample_id tidak valid pada baris {line_number}."
                )

            if sample_id in sample_ids:
                raise ValueError(
                    f"sample_id duplikat ditemukan: {sample_id}"
                )

            for condition in required_conditions:
                result = record.get(condition)

                if not isinstance(result, dict):
                    raise ValueError(
                        f"{sample_id}: kondisi {condition} tidak tersedia."
                    )

                if result.get("sample_id") != sample_id:
                    raise ValueError(
                        f"{sample_id}: sample_id kondisi {condition} "
                        "tidak cocok."
                    )

                response = result.get("response")
                if not isinstance(response, str) or not response.strip():
                    raise ValueError(
                        f"{sample_id}: respons {condition} kosong."
                    )

            sample_ids.add(sample_id)
            record_count += 1

    if record_count == 0:
        raise ValueError(f"Input JSONL kosong: {path}")

    return record_count, sample_ids


def validate_output(
    path: Path,
    expected_count: int,
    expected_ids: set[str],
) -> None:
    """
    Memastikan output klasifikasi ada dan jumlah record cocok.

    Tidak mengasumsikan bahwa semua label harus non-null;
    keputusan classifier tetap perlu diaudit terpisah.
    """
    if not path.exists():
        raise RuntimeError(
            f"Output klasifikasi tidak ditemukan: {path}"
        )

    output_ids = set()
    output_count = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON output tidak valid pada baris {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Output baris {line_number} bukan object JSON."
                )

            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(
                    f"Output baris {line_number} tidak memiliki sample_id."
                )

            if sample_id in output_ids:
                raise ValueError(
                    f"sample_id duplikat pada output: {sample_id}"
                )

            output_ids.add(sample_id)
            output_count += 1

    if output_count != expected_count or output_ids != expected_ids:
        raise RuntimeError(
            "Input dan output tidak cocok.\n"
            f"Input records : {expected_count}\n"
            f"Output records: {output_count}\n"
            f"ID hilang     : {sorted(expected_ids - output_ids)}\n"
            f"ID tambahan   : {sorted(output_ids - expected_ids)}"
        )


def main() -> None:
    print("=" * 70)
    print("ADAPTIVE CLASSIFICATION SMOKE TEST")
    print("=" * 70)

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Folder hasil eksperimen tidak ditemukan: {RESULTS_DIR}"
        )

    input_path = find_latest_input()
    input_count, input_ids = validate_input(input_path)

    output_path = input_path.with_name(
        f"{input_path.stem}_classified.jsonl"
    )

    print(f"\nInput : {input_path}")
    print(f"Output: {output_path}")
    print(f"Samples detected: {input_count}")
    print("Conditions per sample: llm_text, vlm_text, vlm_image")

    classifier = RuleBasedResponseClassifier()

    runner = ClassificationRunner(
        classifier=classifier,
        input_path=input_path,
        output_path=output_path,
    )

    print("\nRunning classification...")
    count = runner.run()
    print(f"Records reported by classifier: {count}")

    validate_output(
        path=output_path,
        expected_count=input_count,
        expected_ids=input_ids,
    )

    print("\n" + "=" * 70)
    print("ADAPTIVE CLASSIFICATION SMOKE TEST PASSED")
    print(f"Input samples validated: {input_count}")
    print(f"Output records validated: {input_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()