from pathlib import Path
from typing import Any

import csv

from datasets.schema import BenchmarkSample


class XSTestAdapter:
    """
    Adapter untuk dataset XSTest.

    Sumber resmi:
        https://github.com/paul-rottger/xstest

    File:
        xstest_prompts.csv

    CSV resmi XSTest memiliki field:
        - id
        - prompt
        - type
        - label
        - focus
        - note

    Adapter ini membaca CSV lokal dan mengubah setiap
    row menjadi BenchmarkSample.

    Data mentah tidak dimodifikasi.
    """

    BENCHMARK_NAME = "XSTest"

    REQUIRED_COLUMNS = {
        "id",
        "prompt",
        "type",
        "label",
        "focus",
        "note",
    }

    DEFAULT_DATA_PATH = (
        Path("data")
        / "raw"
        / "xstest"
        / "xstest_prompts.csv"
    )

    def __init__(
        self,
        data_path: str | Path = DEFAULT_DATA_PATH,
    ):
        """
        Membuat XSTestAdapter.

        Parameters
        ----------
        data_path : str | Path
            Lokasi file xstest_prompts.csv.
        """

        self.data_path = Path(data_path)

    def validate_file(self) -> None:
        """
        Memastikan file dataset tersedia.

        Raises
        ------
        FileNotFoundError
            Jika file tidak ditemukan.
        """

        if not self.data_path.exists():
            raise FileNotFoundError(
                "File XSTest tidak ditemukan: "
                f"{self.data_path}"
            )

        if not self.data_path.is_file():
            raise FileNotFoundError(
                "Path XSTest bukan merupakan file: "
                f"{self.data_path}"
            )

    def load_raw_dataset(self) -> list[dict[str, Any]]:
        """
        Membaca CSV XSTest tanpa melakukan transformasi
        terhadap isi data.

        Returns
        -------
        list[dict[str, Any]]
            Row mentah XSTest.
        """

        self.validate_file()

        with self.data_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(
                    "CSV XSTest tidak memiliki header."
                )

            actual_columns = set(
                reader.fieldnames
            )

            missing_columns = (
                self.REQUIRED_COLUMNS
                - actual_columns
            )

            if missing_columns:
                raise ValueError(
                    "Schema CSV XSTest tidak sesuai "
                    "dengan schema resmi. "
                    f"Kolom yang hilang: "
                    f"{sorted(missing_columns)}. "
                    f"Kolom tersedia: "
                    f"{sorted(actual_columns)}"
                )

            rows = list(reader)

        return rows

    def validate_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        """
        Memvalidasi isi dasar setiap row.

        Validasi ini tidak mengubah data.
        """

        if not rows:
            raise ValueError(
                "Dataset XSTest kosong."
            )

        for index, row in enumerate(rows):

            if not str(
                row.get("id", "")
            ).strip():
                raise ValueError(
                    f"XSTest row {index} "
                    "memiliki id kosong."
                )

            if not str(
                row.get("prompt", "")
            ).strip():
                raise ValueError(
                    f"XSTest row {index} "
                    "memiliki prompt kosong."
                )

            if not str(
                row.get("type", "")
            ).strip():
                raise ValueError(
                    f"XSTest row {index} "
                    "memiliki type kosong."
                )

            if not str(
                row.get("label", "")
            ).strip():
                raise ValueError(
                    f"XSTest row {index} "
                    "memiliki label kosong."
                )

    def convert_sample(
        self,
        row: dict[str, Any],
    ) -> BenchmarkSample:
        """
        Mengubah satu row XSTest menjadi
        BenchmarkSample.

        Parameters
        ----------
        row : dict[str, Any]
            Row asli XSTest.

        Returns
        -------
        BenchmarkSample
            Sample yang sudah dinormalisasi.
        """

        original_id = str(
            row["id"]
        ).strip()

        prompt = str(
            row["prompt"]
        ).strip()

        sample_id = f"xstest_{original_id}"

        category = str(
            row["type"]
        ).strip()

        metadata = {
            "original_id": original_id,
            "label": str(
                row["label"]
            ).strip(),
            "focus": str(
                row["focus"]
            ).strip(),
            "note": str(
                row["note"]
            ).strip(),
            "dataset_name": (
                "paul-rottger/xstest"
            ),
            "source_file": str(
                self.data_path
            ),
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
        Membaca dan menormalisasi seluruh XSTest.

        Returns
        -------
        list[BenchmarkSample]
            List sample XSTest.
        """

        rows = self.load_raw_dataset()

        self.validate_rows(rows)

        samples = []

        for row in rows:
            sample = self.convert_sample(row)

            samples.append(sample)

        return samples

    def __repr__(self) -> str:
        """
        Representasi object untuk debugging.
        """

        return (
            "XSTestAdapter("
            f"data_path='{self.data_path}')"
        )