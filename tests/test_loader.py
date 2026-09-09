from datasets.loader import DatasetLoader


def test_supported_benchmarks():
    """
    Memastikan seluruh benchmark penelitian
    terdaftar.
    """

    loader = DatasetLoader("jbb")

    expected = [
        "advbench",
        "harmbench",
        "jbb",
        "xstest",
    ]

    actual = loader.get_supported_benchmarks()

    assert actual == expected


def test_benchmark_name_is_normalized():
    """
    Nama benchmark harus dinormalisasi menjadi
    lowercase dan tanpa whitespace di luar.
    """

    loader = DatasetLoader("  XSTEST  ")

    assert loader.benchmark == "xstest"


def test_invalid_benchmark():
    """
    Benchmark yang tidak dikenal harus ditolak.
    """

    try:
        DatasetLoader("unknown_dataset")

    except ValueError:
        return

    raise AssertionError(
        "DatasetLoader seharusnya menolak "
        "benchmark yang tidak didukung."
    )


def test_jbb_adapter_is_selected():
    """
    Memastikan JBBAdapter dipilih untuk JBB.
    """

    loader = DatasetLoader("jbb")

    assert (
        loader.adapter.__class__.__name__
        == "JBBAdapter"
    )


def test_advbench_adapter_is_selected():
    """
    Memastikan AdvBenchAdapter dipilih
    untuk AdvBench.
    """

    loader = DatasetLoader("advbench")

    assert (
        loader.adapter.__class__.__name__
        == "AdvBenchAdapter"
    )


def test_harmbench_adapter_is_selected():
    """
    Memastikan HarmBenchAdapter dipilih
    untuk HarmBench.
    """

    loader = DatasetLoader("harmbench")

    assert (
        loader.adapter.__class__.__name__
        == "HarmBenchAdapter"
    )


def test_xstest_adapter_is_selected():
    """
    Memastikan XSTestAdapter dipilih
    untuk XSTest.
    """

    loader = DatasetLoader("xstest")

    assert (
        loader.adapter.__class__.__name__
        == "XSTestAdapter"
    )


def test_harmbench_config_is_passed():
    """
    Memastikan konfigurasi HarmBench diteruskan
    ke adapter.
    """

    loader = DatasetLoader(
        "harmbench",
        config="contextual",
    )

    assert (
        loader.adapter.config
        == "contextual"
    )


def test_harmbench_split_is_passed():
    """
    Memastikan split HarmBench diteruskan
    ke adapter.
    """

    loader = DatasetLoader(
        "harmbench",
        split="train",
    )

    assert (
        loader.adapter.split
        == "train"
    )


def test_loader_representation_jbb():
    """
    Memastikan representasi JBB benar.
    """

    loader = DatasetLoader("jbb")

    expected = (
        "DatasetLoader("
        "benchmark='jbb', "
        "adapter=JBBAdapter)"
    )

    assert repr(loader) == expected


def test_loader_representation_advbench():
    """
    Memastikan representasi AdvBench benar.
    """

    loader = DatasetLoader("advbench")

    expected = (
        "DatasetLoader("
        "benchmark='advbench', "
        "adapter=AdvBenchAdapter)"
    )

    assert repr(loader) == expected


def test_loader_representation_harmbench():
    """
    Memastikan representasi HarmBench benar.
    """

    loader = DatasetLoader("harmbench")

    expected = (
        "DatasetLoader("
        "benchmark='harmbench', "
        "adapter=HarmBenchAdapter)"
    )

    assert repr(loader) == expected


def test_loader_representation_xstest():
    """
    Memastikan representasi XSTest benar.
    """

    loader = DatasetLoader("xstest")

    expected = (
        "DatasetLoader("
        "benchmark='xstest', "
        "adapter=XSTestAdapter)"
    )

    assert repr(loader) == expected


def main():
    test_supported_benchmarks()
    test_benchmark_name_is_normalized()
    test_invalid_benchmark()

    test_jbb_adapter_is_selected()
    test_advbench_adapter_is_selected()
    test_harmbench_adapter_is_selected()
    test_xstest_adapter_is_selected()

    test_harmbench_config_is_passed()
    test_harmbench_split_is_passed()

    test_loader_representation_jbb()
    test_loader_representation_advbench()
    test_loader_representation_harmbench()
    test_loader_representation_xstest()

    print(
        "Semua DatasetLoader test berhasil."
    )


if __name__ == "__main__":
    main()