from typing import Any

from datasets import load_dataset

from datasets.schema import BenchmarkSample


class AdvBenchAdapter:
    """
    Adapter untuk dataset AdvBench.

    Sumber resmi yang digunakan dalam project:
        walledai/AdvBench

    Hugging Face:
        https://huggingface.co/datasets/walledai/AdvBench

    Adapter ini bertugas:
        1. Memuat dataset AdvBench.
        2. Memeriksa schema dataset.
        3. Menentukan kolom prompt secara eksplisit.
        4. Mengubah setiap sample menjadi BenchmarkSample.

    Adapter tidak mengubah data asli benchmark.
    """

    DATASET_NAME = "walledai/AdvBench"

    BENCHMARK_NAME = "AdvBench"

    def __init__(
        self,
        split: str = "train",
    ):
        """
        Membuat AdvBenchAdapter.

        Parameters
        ----------
        split : str
            Split dataset yang akan digunakan.
        """

        self.split = split

    def load_raw_dataset(self):
        """
        Memuat dataset asli AdvBench dari Hugging Face.

        Returns
        -------
        datasets.Dataset
            Dataset AdvBench sebelum normalisasi.
        """

        dataset = load_dataset(
            self.DATASET_NAME,
            split=self.split,
        )

        return dataset

    def _detect_prompt_column(self, dataset) -> str:
        """
        Mendeteksi kolom yang kemungkinan merupakan
        kolom instruction/prompt AdvBench.

        Kita tidak langsung mengasumsikan nama kolom,
        tetapi memeriksa schema aktual dataset.

        Returns
        -------
        str
            Nama kolom prompt.

        Raises
        ------
        ValueError
            Jika kolom prompt tidak dapat ditentukan
            secara aman.
        """

        columns = set(dataset.column_names)

        candidate_columns = [
            "goal",
            "prompt",
            "instruction",
            "text",
        ]

        available_candidates = [
            column
            for column in candidate_columns
            if column in columns
        ]

        if len(available_candidates) == 1:
            return available_candidates[0]

        if len(available_candidates) > 1:
            raise ValueError(
                "Dataset AdvBench memiliki lebih dari satu "
                "kolom yang mungkin digunakan sebagai prompt: "
                f"{available_candidates}. "
                "Tentukan kolom secara eksplisit."
            )

        raise ValueError(
            "Tidak dapat menemukan kolom prompt AdvBench. "
            f"Kolom yang tersedia: {sorted(columns)}"
        )

    def validate_columns(self, dataset) -> str:
        """
        Memvalidasi schema dataset dan mengembalikan
        nama kolom prompt.

        Parameters
        ----------
        dataset : datasets.Dataset
            Dataset AdvBench.

        Returns
        -------
        str
            Nama kolom prompt.
        """

        if not dataset.column_names:
            raise ValueError(
                "Dataset AdvBench tidak memiliki kolom."
            )

        prompt_column = self._detect_prompt_column(dataset)

        return prompt_column

    def convert_sample(
        self,
        row: dict[str, Any],
        index: int,
        prompt_column: str,
    ) -> BenchmarkSample:
        """
        Mengubah satu row AdvBench menjadi BenchmarkSample.

        Parameters
        ----------
        row : dict
            Row asli dari AdvBench.

        index : int
            Posisi row dalam dataset.

        prompt_column : str
            Nama kolom yang digunakan sebagai prompt.

        Returns
        -------
        BenchmarkSample
            Sample dalam format internal pipeline.
        """

        prompt = str(row[prompt_column]).strip()

        if not prompt:
            raise ValueError(
                f"AdvBench sample pada index {index} "
                "memiliki prompt kosong."
            )

        sample_id = f"advbench_{index:04d}"

        metadata = {
            "original_index": index,
            "prompt_column": prompt_column,
            "dataset_name": self.DATASET_NAME,
            "dataset_split": self.split,
        }

        for key, value in row.items():
            if key != prompt_column:
                metadata[key] = value

        return BenchmarkSample(
            sample_id=sample_id,
            benchmark=self.BENCHMARK_NAME,
            prompt=prompt,
            category=None,
            metadata=metadata,
        )

    def load(self) -> list[BenchmarkSample]:
        """
        Memuat AdvBench dan menormalisasi seluruh sample
        menjadi BenchmarkSample.

        Returns
        -------
        list[BenchmarkSample]
            Sample AdvBench yang telah dinormalisasi.
        """

        dataset = self.load_raw_dataset()

        prompt_column = self.validate_columns(dataset)

        samples = []

        for index, row in enumerate(dataset):
            sample = self.convert_sample(
                row=row,
                index=index,
                prompt_column=prompt_column,
            )

            samples.append(sample)

        return samples

    def __repr__(self) -> str:
        """
        Representasi object untuk debugging.
        """

        return (
            "AdvBenchAdapter("
            f"dataset='{self.DATASET_NAME}', "
            f"split='{self.split}')"
        )