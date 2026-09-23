import json
from pathlib import Path
from typing import Iterable

from preprocessing.manifest import VisualSample


class VisualManifestWriter:
    """
    Menulis VisualSample ke dalam format JSONL.

    Satu VisualSample = satu baris JSON.
    """

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

    def write(
        self,
        samples: Iterable[VisualSample],
    ) -> Path:
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for sample in samples:
                record = sample.to_dict()

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                )
                file.write("\n")

        return self.output_path