from benchmark_data.adapters.harmbench import HarmBenchAdapter

def test_invalid_config():
    """
    Config yang tidak tersedia harus ditolak.
    """

    try:
        HarmBenchAdapter(
            config="unknown"
        )

    except ValueError:
        return

    raise AssertionError(
        "HarmBenchAdapter seharusnya menolak "
        "config yang tidak didukung."
    )


def test_supported_configs():
    """
    Memastikan seluruh config resmi tersedia.
    """

    adapter = HarmBenchAdapter(
        config="standard"
    )

    expected = {
        "standard",
        "contextual",
        "copyright",
    }

    assert (
        adapter.SUPPORTED_CONFIGS
        == expected
    )


def test_standard():
    """
    Menguji config standard.
    """

    adapter = HarmBenchAdapter(
        config="standard"
    )

    samples = adapter.load()

    assert len(samples) > 0

    sample = samples[0]

    assert sample.benchmark == "HarmBench"
    assert sample.prompt
    assert sample.category
    assert (
        sample.metadata["dataset_config"]
        == "standard"
    )


def test_contextual():
    """
    Menguji config contextual.
    """

    adapter = HarmBenchAdapter(
        config="contextual"
    )

    samples = adapter.load()

    assert len(samples) > 0

    sample = samples[0]

    assert sample.benchmark == "HarmBench"
    assert sample.prompt
    assert sample.category

    assert "context" in sample.metadata

    assert (
        sample.metadata["dataset_config"]
        == "contextual"
    )


def test_copyright():
    """
    Menguji config copyright.
    """

    adapter = HarmBenchAdapter(
        config="copyright"
    )

    samples = adapter.load()

    assert len(samples) > 0

    sample = samples[0]

    assert sample.benchmark == "HarmBench"
    assert sample.prompt

    assert (
        sample.category is None
    )

    assert "tags" in sample.metadata

    assert (
        sample.metadata["dataset_config"]
        == "copyright"
    )


def main():
    print(
        "Menjalankan HarmBenchAdapter tests..."
    )

    test_invalid_config()

    print(
        "✓ Invalid config test"
    )

    test_supported_configs()

    print(
        "✓ Supported config test"
    )

    print(
        "\nMemuat HarmBench standard..."
    )

    test_standard()

    print(
        "✓ Standard config"
    )

    print(
        "\nMemuat HarmBench contextual..."
    )

    test_contextual()

    print(
        "✓ Contextual config"
    )

    print(
        "\nMemuat HarmBench copyright..."
    )

    test_copyright()

    print(
        "✓ Copyright config"
    )

    print(
        "\nSemua HarmBenchAdapter "
        "test berhasil."
    )


if __name__ == "__main__":
    main()