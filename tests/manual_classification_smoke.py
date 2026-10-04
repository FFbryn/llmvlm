from pathlib import Path

from evaluation.classification import (
    ClassificationRunner,
    RuleBasedResponseClassifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "smoke_test"
    / "jbb_real_batch.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "benchmark_data"
    / "results"
    / "smoke_test"
    / "jbb_real_batch_classified.jsonl"
)


def main():
    print("=" * 70)
    print("CLASSIFICATION SMOKE TEST")
    print("=" * 70)

    print(f"\nInput : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    classifier = RuleBasedResponseClassifier()

    runner = ClassificationRunner(
        classifier=classifier,
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
    )

    count = runner.run()

    print(f"\nRecords classified: {count}")

    if count == 0:
        raise RuntimeError(
            "Tidak ada record yang berhasil diklasifikasikan."
        )

    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            "Classified output tidak ditemukan."
        )

    print("\n" + "=" * 70)
    print("CLASSIFICATION SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()