import json
from pathlib import Path
from typing import Any


class ClassifiedResultWriter:
    """
    Menulis hasil classification ke JSONL.

    Writer ini tidak mengubah raw input.
    """

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write(self, record: dict[str, Any]) -> None:
        if not isinstance(record, dict):
            raise TypeError(
                "record harus berupa dictionary."
            )

        with self.output_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    def write_many(
        self,
        records,
    ) -> None:
        for record in records:
            self.write(record)