from pathlib import Path

from preprocessing.build_visual_dataset import (
    build_visual_dataset,
)


def main():
    output_dir = Path(
        "benchmark_data/generated/text_rendered"
    )

    manifest_path = Path(
        "benchmark_data/generated/"
        "visual_manifest_jbb.jsonl"
    )

    result = build_visual_dataset(
        benchmark="jbb",
        output_dir=output_dir,
        manifest_path=manifest_path,
    )

    print()
    print("Visual dataset selesai dibuat.")
    print(f"Manifest: {result}")


if __name__ == "__main__":
    main()