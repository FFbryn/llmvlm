from typing import Any

from benchmark_data.schema import BenchmarkSample
from benchmark_data.adapters.jbb import JBBAdapter
from benchmark_data.adapters.advbench import AdvBenchAdapter
from benchmark_data.adapters.harmbench import HarmBenchAdapter
from benchmark_data.adapters.xstest import XSTestAdapter


class DatasetLoader:
    """
    Entry point utama untuk memuat benchmark penelitian.

    DatasetLoader bertugas memilih adapter berdasarkan
    nama benchmark.

    Detail loading dan schema masing-masing benchmark
    ditangani oleh adapter masing-masing.

    Benchmark yang didukung:
        - jbb
        - advbench
        - harmbench
        - xstest
    """

    SUPPORTED_BENCHMARKS = {
        "jbb",
        "advbench",
        "harmbench",
        "xstest",
    }

    def __init__(
        self,
        benchmark: str,
        **adapter_kwargs: Any,
    ):
        """
        Membuat DatasetLoader.

        Parameters
        ----------
        benchmark : str
            Nama benchmark.

        adapter_kwargs : Any
            Parameter tambahan yang diteruskan
            ke adapter.
        """

        benchmark = benchmark.lower().strip()

        if benchmark not in self.SUPPORTED_BENCHMARKS:
            supported = ", ".join(
                sorted(self.SUPPORTED_BENCHMARKS)
            )

            raise ValueError(
                f"Benchmark '{benchmark}' tidak didukung. "
                f"Benchmark yang tersedia: {supported}"
            )

        self.benchmark = benchmark
        self.adapter_kwargs = adapter_kwargs

        self.adapter = self._create_adapter()

    def _create_adapter(self):
        """
        Membuat adapter berdasarkan benchmark.
        """

        if self.benchmark == "jbb":
            return JBBAdapter(
                **self.adapter_kwargs
            )

        if self.benchmark == "advbench":
            return AdvBenchAdapter(
                **self.adapter_kwargs
            )

        if self.benchmark == "harmbench":
            return HarmBenchAdapter(
                **self.adapter_kwargs
            )

        if self.benchmark == "xstest":
            return XSTestAdapter(
                **self.adapter_kwargs
            )

        raise NotImplementedError(
            f"Adapter untuk benchmark "
            f"'{self.benchmark}' "
            "belum diimplementasikan."
        )

    def load(self) -> list[BenchmarkSample]:
        """
        Memuat dataset menggunakan adapter
        yang telah dipilih.

        Returns
        -------
        list[BenchmarkSample]
            Dataset yang telah dinormalisasi.
        """

        return self.adapter.load()

    def get_supported_benchmarks(self) -> list[str]:
        """
        Mengembalikan seluruh benchmark yang
        didukung DatasetLoader.
        """

        return sorted(
            self.SUPPORTED_BENCHMARKS
        )

    def __repr__(self) -> str:
        """
        Representasi object untuk debugging.
        """

        return (
            f"DatasetLoader("
            f"benchmark='{self.benchmark}', "
            f"adapter="
            f"{self.adapter.__class__.__name__})"
        )