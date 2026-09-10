from benchmark_data.adapters.advbench import AdvBenchAdapter


def main():
    print("Membuat AdvBenchAdapter...")

    adapter = AdvBenchAdapter()

    print(adapter)

    print("\nMemuat AdvBench dari Hugging Face...")

    samples = adapter.load()

    print("AdvBench berhasil dimuat.")

    print(f"Jumlah sample: {len(samples)}")

    assert len(samples) > 0

    sample = samples[0]

    print("\n===== SAMPLE PERTAMA =====")

    print(f"Sample ID : {sample.sample_id}")
    print(f"Benchmark : {sample.benchmark}")
    print(f"Category  : {sample.category}")
    print(f"Prompt    : {sample.prompt}")

    print("\n===== METADATA =====")

    for key, value in sample.metadata.items():
        print(f"{key}: {value}")

    assert sample.benchmark == "AdvBench"
    assert sample.prompt
    assert sample.sample_id.startswith("advbench_")
    assert "original_index" in sample.metadata
    assert "prompt_column" in sample.metadata
    assert "dataset_name" in sample.metadata

    print("\nAdvBenchAdapter test berhasil.")


if __name__ == "__main__":
    main()