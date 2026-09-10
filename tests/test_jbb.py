from benchmark_data.adapters.jbb import JBBAdapter


def main():
    print("Membuat JBBAdapter...")

    adapter = JBBAdapter()

    print(adapter)

    print("\nMemuat dataset JBB dari Hugging Face...")

    samples = adapter.load()

    print("Dataset berhasil dimuat.")

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

    assert sample.benchmark == "JBB-Behaviors"
    assert sample.prompt
    assert sample.category
    assert "target" in sample.metadata
    assert "behavior" in sample.metadata
    assert "source" in sample.metadata

    print("\nJBBAdapter test berhasil.")


if __name__ == "__main__":
    main()