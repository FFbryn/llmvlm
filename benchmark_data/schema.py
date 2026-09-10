from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BenchmarkSample:
    """
    Representasi standar satu test case dari benchmark.

    Class ini digunakan sebagai format internal yang seragam
    untuk JBB-Behaviors, AdvBench, HarmBench, dan XSTest.

    Data asli benchmark tidak diubah.
    Adapter masing-masing benchmark akan mengubah data asli
    menjadi objek BenchmarkSample.
    """

    # Identitas unik test case dalam pipeline kita
    sample_id: str

    # Nama benchmark asal
    benchmark: str

    # Teks yang akan diberikan kepada model
    prompt: str

    # Kategori benchmark, jika tersedia
    category: Optional[str] = None

    # Metadata tambahan dari dataset asli
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Mengubah BenchmarkSample menjadi dictionary.

        Berguna ketika hasil dataset akan disimpan
        ke JSON atau diproses lebih lanjut.
        """
        return {
            "sample_id": self.sample_id,
            "benchmark": self.benchmark,
            "prompt": self.prompt,
            "category": self.category,
            "metadata": self.metadata,
        }