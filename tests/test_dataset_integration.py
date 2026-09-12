from benchmark_data.loader import DatasetLoader


def test_jbb_integration():
    """
    Menguji seluruh pipeline loading JBB.
    """

    loader = DatasetLoader("jbb")

    samples = loader.load()

    assert len(samples) > 0

    sample = samples[0]

    assert sample.benchmark == "JBB-Behaviors"
    assert sample.prompt
    assert sample.sample_id


def test_advbench_integration():
    """
    Menguji seluruh pipeline loading AdvBench.
    """

    loader = DatasetLoader("advbench")

    samples = loader.load()

    assert len(samples) > 0

    sample = samples[0]

    assert sample.benchmark == "AdvBench"
    assert sample.prompt
    assert sample.sample_id


def test_harmbench_standard_integration():
    """
    Menguji seluruh pipeline loading HarmBench
    standard.
    """

    loader = DatasetLoader(
        "harmbench",
        config="standard",
    )

    samples = loader.load()

    assert len(samples) > 0

    sample = samples[0]

    assert sample.benchmark == "HarmBench"
    assert sample.prompt
    assert sample.sample_id
    assert sample.category


def test_harmbench_contextual_integration():
    """
    Menguji seluruh pipeline loading HarmBench
    contextual.
    """

    loader = DatasetLoader(
        "harmbench",
        config="contextual",
    )

    samples = loader.load()

    assert len(samples) > 0

    sample = samples[0]

    assert sample.benchmark == "HarmBench"
    assert sample.prompt
    assert sample.sample_id
    assert sample.category

    assert "context" in sample.metadata


def test_harmbench_copyright_integration():
    """
    Menguji seluruh pipeline loading HarmBench
    copyright.
    """

    loader = DatasetLoader(
        "harmbench",
        config="copyright",
    )

    samples = loader.load()

    assert len(samples) > 0

    sample = samples[0]

    assert sample.benchmark == "HarmBench"
    assert sample.prompt
    assert sample.sample_id

    assert "tags" in sample.metadata


def test_xstest_integration():
    """
    Menguji seluruh pipeline loading XSTest.
    """

    loader = DatasetLoader("xstest")

    samples = loader.load()

    assert len(samples) > 0

    sample = samples[0]

    assert sample.benchmark == "XSTest"
    assert sample.prompt
    assert sample.sample_id
    assert sample.category

    assert "original_id" in sample.metadata
    assert "label" in sample.metadata
    assert "focus" in sample.metadata
    assert "note" in sample.metadata


def main():
    print(
        "========================================"
    )

    print(
        "DATASET INTEGRATION TEST"
    )

    print(
        "========================================"
    )

    print(
        "\n[1/6] Testing JBB..."
    )

    test_jbb_integration()

    print(
        "✓ JBB integration berhasil."
    )

    print(
        "\n[2/6] Testing AdvBench..."
    )

    test_advbench_integration()

    print(
        "✓ AdvBench integration berhasil."
    )

    print(
        "\n[3/6] Testing HarmBench standard..."
    )

    test_harmbench_standard_integration()

    print(
        "✓ HarmBench standard berhasil."
    )

    print(
        "\n[4/6] Testing HarmBench contextual..."
    )

    test_harmbench_contextual_integration()

    print(
        "✓ HarmBench contextual berhasil."
    )

    print(
        "\n[5/6] Testing HarmBench copyright..."
    )

    test_harmbench_copyright_integration()

    print(
        "✓ HarmBench copyright berhasil."
    )

    print(
        "\n[6/6] Testing XSTest..."
    )

    test_xstest_integration()

    print(
        "✓ XSTest integration berhasil."
    )

    print(
        "\n========================================"
    )

    print(
        "SEMUA DATASET INTEGRATION TEST BERHASIL"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()