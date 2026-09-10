from typing import Any

from datasets import load_dataset

from benchmark_data.schema import BenchmarkSample


class HarmBenchAdapter:
    """
    Adapter untuk dataset HarmBench.

    Sumber:
        walledai/HarmBench

    Dataset memiliki tiga config:
        - standard
        - contextual
        - copyright

    Setiap config menggunakan split:
        train

    Adapter ini tidak mengubah dataset asli.
    Adapter hanya menormalisasi setiap row menjadi
    BenchmarkSample.
    """

    DATASET_NAME = "walledai/HarmBench"

    BENCHMARK_NAME = "HarmBench"

    SUPPORTED_CONFIGS = {
        "standard",
        "contextual",
        "copyright",
    }

    DEFAULT_CONFIG = "standard"
    DEFAULT_SPLIT = "train"

    def __init__(
        self,
        config: str = DEFAULT_CONFIG,
        split: str = DEFAULT_SPLIT,
    ):
        """
        Membuat HarmBenchAdapter.

        Parameters
        ----------
        config : str
            Config HarmBench yang digunakan.

            Pilihan:
                - standard
                - contextual
                - copyright

        split : str
            Split dataset.

            Default:
                train
        """

        config = config.lower().strip()

        if config not in self.SUPPORTED_CONFIGS:
            supported = ", ".join(
                sorted(self.SUPPORTED_CONFIGS)
            )

            raise ValueError(
                f"Config HarmBench '{config}' tidak didukung. "
                f"Config yang tersedia: {supported}"
            )

        self.config = config
        self.split = split

    def load_raw_dataset(self):
        """
        Memuat dataset HarmBench asli dari Hugging Face.

        Returns
        -------
        datasets.Dataset
            Dataset sebelum normalisasi.
        """

        dataset = load_dataset(
            self.DATASET_NAME,
            self.config,
            split=self.split,
        )

        return dataset

    def validate_columns(self, dataset) -> None:
        """
        Memastikan schema dataset sesuai dengan config
        yang sedang digunakan.

        Parameters
        ----------
        dataset : datasets.Dataset
            Dataset HarmBench.

        Raises
        ------
        ValueError
            Jika kolom yang diperlukan tidak tersedia.
        """

        columns = set(dataset.column_names)

        required_columns = {
            "prompt",
        }

        if self.config in {
            "standard",
            "contextual",
        }:
            required_columns.add("category")

        if self.config == "contextual":
            required_columns.add("context")

        if self.config == "copyright":
            required_columns.add("tags")

        missing_columns = required_columns - columns

        if missing_columns:
            raise ValueError(
                "Schema HarmBench tidak sesuai dengan config "
                f"'{self.config}'. "
                f"Kolom yang hilang: {sorted(missing_columns)}. "
                f"Kolom yang tersedia: {sorted(columns)}"
            )

    def convert_sample(
        self,
        row: dict[str, Any],
        index: int,
    ) -> BenchmarkSample:
        """
        Mengubah satu row HarmBench menjadi
        BenchmarkSample.

        Parameters
        ----------
        row : dict
            Row asli dari HarmBench.

        index : int
            Posisi row dalam dataset.

        Returns
        -------
        BenchmarkSample
            Sample dalam format internal pipeline.
        """

        prompt = str(row["prompt"]).strip()

        if not prompt:
            raise ValueError(
                f"HarmBench sample index {index} "
                "memiliki prompt kosong."
            )

        sample_id = (
            f"harmbench_"
            f"{self.config}_"
            f"{index:04d}"
        )

        category = None

        if "category" in row:
            category = str(
                row["category"]
            ).strip()

        metadata = {
            "original_index": index,
            "dataset_name": self.DATASET_NAME,
            "dataset_config": self.config,
            "dataset_split": self.split,
        }

        if "context" in row:
            metadata["context"] = row["context"]

        if "tags" in row:
            metadata["tags"] = row["tags"]

        return BenchmarkSample(
            sample_id=sample_id,
            benchmark=self.BENCHMARK_NAME,
            prompt=prompt,
            category=category,
            metadata=metadata,
        )

    def load(self) -> list[BenchmarkSample]:
        """
        Memuat dan menormalisasi seluruh sample
        HarmBench.

        Returns
        -------
        list[BenchmarkSample]
            List BenchmarkSample.
        """

        dataset = self.load_raw_dataset()

        self.validate_columns(dataset)

        samples = []

        for index, row in enumerate(dataset):
            sample = self.convert_sample(
                row=row,
                index=index,
            )

            samples.append(sample)

        return samples

    def __repr__(self) -> str:
        """
        Representasi object untuk debugging.
        """

        return (
            "HarmBenchAdapter("
            f"dataset='{self.DATASET_NAME}', "
            f"config='{self.config}', "
            f"split='{self.split}')"
        )