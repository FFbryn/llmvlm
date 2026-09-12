from benchmark_data.schema import BenchmarkSample


def main():
    sample = BenchmarkSample(
        sample_id="test_001",
        benchmark="TEST",
        prompt="Ini adalah test prompt.",
        category="test_category",
        metadata={
            "source": "unit_test"
        }
    )

    print("BenchmarkSample berhasil dibuat.")

    print("\n===== SAMPLE =====")
    print(sample)

    print("\n===== DICTIONARY =====")
    print(sample.to_dict())

    assert sample.sample_id == "test_001"
    assert sample.benchmark == "TEST"
    assert sample.prompt == "Ini adalah test prompt."
    assert sample.category == "test_category"
    assert sample.metadata["source"] == "unit_test"

    print("\nSchema test berhasil.")


if __name__ == "__main__":
    main()