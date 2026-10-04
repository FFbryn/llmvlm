import json
from pathlib import Path
from typing import Any, Iterator


class RawExperimentResultLoader:
    """
    Membaca raw experiment result dari JSONL.

    Loader ini tidak melakukan classification.
    Tugasnya hanya membaca dan memvalidasi struktur dasar record.
    """

    def __init__(self, input_path: Path):
        self.input_path = Path(input_path)

    def _validate_record(
        self,
        record: dict[str, Any],
        line_number: int,
    ) -> None:
        if not isinstance(record, dict):
            raise ValueError(
                f"Record pada line {line_number} harus berupa object JSON."
            )

        sample_id = record.get("sample_id")

        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(
                f"sample_id tidak valid pada line {line_number}."
            )

    def load(self) -> list[dict[str, Any]]:
        """
        Membaca seluruh record dari JSONL.
        """
        return list(self.iter_records())

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """
        Membaca record satu per satu.

        Generator digunakan agar file besar tidak harus
        dimuat seluruhnya ke memory.
        """
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Raw result tidak ditemukan: {self.input_path}"
            )

        with self.input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON tidak valid pada line "
                        f"{line_number}: {exc}"
                    ) from exc

                self._validate_record(
                    record,
                    line_number,
                )

                yield record