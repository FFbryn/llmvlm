import json
from pathlib import Path

from preprocessing.manifest import VisualSample


class VisualManifestLoader:
    """
    Membaca visual_manifest.jsonl dan mengubah setiap
    record menjadi VisualSample.
    """

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)

    def load(self) -> list[VisualSample]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest tidak ditemukan: "
                f"{self.manifest_path}"
            )

        samples: list[VisualSample] = []

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON tidak valid pada "
                        f"baris {line_number}: "
                        f"{self.manifest_path}"
                    ) from exc

                required_fields = {
                    "sample_id",
                    "benchmark",
                    "image_path",
                }

                missing = (
                    required_fields
                    - record.keys()
                )

                if missing:
                    raise ValueError(
                        f"Field wajib hilang pada "
                        f"baris {line_number}: "
                        f"{sorted(missing)}"
                    )

                samples.append(
                    VisualSample(
                        sample_id=record["sample_id"],
                        benchmark=record["benchmark"],
                        image_path=Path(
                            record["image_path"]
                        ),
                        source_prompt=record.get(
                            "source_prompt"
                        ),
                        rendering_config=record.get(
                            "rendering_config"
                        ),
                    )
                )

        return samples