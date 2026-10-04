import json
from pathlib import Path
from typing import Any

from evaluation.three_condition_result import (
    ThreeConditionResult,
)


class RawExperimentResultWriter:
    """
    Writer untuk menyimpan hasil mentah eksperimen.

    Satu ThreeConditionResult disimpan sebagai satu
    JSON object dalam satu baris JSONL.

    Writer ini tidak melakukan:
        - classification
        - refusal detection
        - transferability analysis
        - statistical analysis

    Tanggung jawabnya hanya persistence raw result.
    """

    def __init__(
        self,
        output_path: Path,
    ):
        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _serialize_condition(
        self,
        result,
    ) -> dict[str, Any] | None:
        """
        Mengubah ExperimentResult menjadi dictionary.

        None tetap dipertahankan jika suatu condition
        belum tersedia.
        """

        if result is None:
            return None

        return result.to_dict()

    def serialize(
        self,
        result: ThreeConditionResult,
    ) -> dict[str, Any]:
        """
        Mengubah ThreeConditionResult menjadi dictionary
        yang siap disimpan sebagai JSON.
        """

        if result is None:
            raise ValueError(
                "ThreeConditionResult tidak boleh None."
            )

        data = {
            "sample_id": result.sample_id,
            "llm_text": self._serialize_condition(
                result.llm_text
            ),
            "vlm_text": self._serialize_condition(
                result.vlm_text
            ),
            "vlm_image": self._serialize_condition(
                result.vlm_image
            ),
        }

        return data

    def write(
        self,
        result: ThreeConditionResult,
    ) -> None:
        """
        Menambahkan satu ThreeConditionResult
        ke file JSONL.

        Mode append digunakan agar hasil sample
        sebelumnya tidak tertimpa.
        """

        data = self.serialize(result)

        with self.output_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    default=str,
                )
            )
            file.write("\n")

    def write_many(
        self,
        results: list[ThreeConditionResult],
    ) -> None:
        """
        Menyimpan beberapa hasil eksperimen.

        Setiap result tetap disimpan sebagai satu
        JSON object per baris.
        """

        for result in results:
            self.write(result)