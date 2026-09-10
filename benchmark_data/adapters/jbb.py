from typing import Optional

from datasets import load_dataset

from benchmark_data.schema import BenchmarkSample


class JBBAdapter:
    """
    Adapter untuk dataset JBB-Behaviors.

    Sumber:
        JailbreakBench/JBB-Behaviors

    Repository:
        https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors

    Dataset digunakan dalam bentuk aslinya dari Hugging Face.
    Adapter ini hanya melakukan normalisasi ke BenchmarkSample.
    """

    DATASET_NAME = "JailbreakBench/JBB-Behaviors"
    DATASET_CONFIG = "behaviors"
    DATASET_SPLIT = "harmful"

    BENCHMARK_NAME = "JBB-Behaviors"

    def __init__(
        self,
        split: str = DATASET_SPLIT,
        config: str = DATASET_CONFIG,
    ):
        """
        Membuat JBBAdapter.

        Parameters
        ----------
        split : str
            Split dataset yang digunakan.

        config : str
            Konfigurasi/subset dataset.
        """

        self.split = split
        self.config = config

    def load_raw_dataset(self):
        """
        Memuat dataset asli dari Hugging Face.

        Returns
        -------
        datasets.Dataset
            Dataset asli sebelum dinormalisasi.
        """

        dataset = load_dataset(
            self.DATASET_NAME,
            self.config,
            split=self.split,
        )

        return dataset

    def validate_columns(self, dataset) -> None:
        """
        Memastikan dataset JBB mempunyai kolom
        yang dibutuhkan oleh adapter.

        Parameters
        ----------
        dataset : datasets.Dataset
            Dataset JBB yang sudah dimuat.

        Raises
        ------
        ValueError
            Jika terdapat kolom yang dibutuhkan tetapi tidak tersedia.
        """

        required_columns = {
            "Index",
            "Goal",
            "Target",
            "Behavior",
            "Category",
            "Source",
        }

        actual_columns = set(dataset.column_names)

        missing_columns = required_columns - actual_columns

        if missing_columns:
            raise ValueError(
                "Dataset JBB tidak memiliki kolom yang diperlukan. "
                f"Kolom yang hilang: {sorted(missing_columns)}"
            )

    def convert_sample(self, row) -> BenchmarkSample:
        """
        Mengubah satu row JBB menjadi BenchmarkSample.

        Parameters
        ----------
        row : dict
            Satu row dari dataset JBB.

        Returns
        -------
        BenchmarkSample
            Sample dalam format internal pipeline.
        """

        sample_id = f"jbb_{row['Index']}"

        prompt = str(row["Goal"]).strip()

        category = str(row["Category"]).strip()

        metadata = {
            "original_index": row["Index"],
            "behavior": row["Behavior"],
            "target": row["Target"],
            "source": row["Source"],
            "dataset_name": self.DATASET_NAME,
            "dataset_config": self.config,
            "dataset_split": self.split,
        }

        return BenchmarkSample(
            sample_id=sample_id,
            benchmark=self.BENCHMARK_NAME,
            prompt=prompt,
            category=category,
            metadata=metadata,
        )

    def load(self) -> list[BenchmarkSample]:
        """
        Memuat JBB dan mengubah seluruh sample
        menjadi BenchmarkSample.

        Returns
        -------
        list[BenchmarkSample]
            Sample JBB yang sudah dinormalisasi.
        """

        dataset = self.load_raw_dataset()

        self.validate_columns(dataset)

        samples = []

        for row in dataset:
            sample = self.convert_sample(row)
            samples.append(sample)

        return samples

    def __repr__(self) -> str:
        """
        Representasi object untuk debugging.
        """

        return (
            "JBBAdapter("
            f"dataset='{self.DATASET_NAME}', "
            f"config='{self.config}', "
            f"split='{self.split}')"
        )