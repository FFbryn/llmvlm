from pathlib import Path
from typing import Any

from evaluation.classification.base import BaseResponseClassifier
from evaluation.classification.classified_result_writer import (
    ClassifiedResultWriter,
)
from evaluation.classification.labels import ClassificationLabel
from evaluation.classification.raw_result_loader import (
    RawExperimentResultLoader,
)


class ClassificationRunner:
    """
    Menjalankan classification terhadap raw experiment result.

    Pipeline:

        raw JSONL
            ↓
        load record
            ↓
        classify response
            ↓
        classified JSONL

    Model tidak dijalankan di sini.
    """

    CONDITIONS = (
        "llm_text",
        "vlm_text",
        "vlm_image",
    )

    def __init__(
        self,
        *,
        classifier: BaseResponseClassifier,
        input_path: Path,
        output_path: Path,
    ):
        if not isinstance(
            classifier,
            BaseResponseClassifier,
        ):
            raise TypeError(
                "classifier harus merupakan "
                "instance BaseResponseClassifier."
            )

        self.classifier = classifier
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        self.loader = RawExperimentResultLoader(
            self.input_path
        )

        self.writer = ClassifiedResultWriter(
            self.output_path
        )

    def _classify_condition(
        self,
        condition: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if condition is None:
            return None

        if not isinstance(condition, dict):
            raise ValueError(
                "Condition result harus berupa dictionary "
                "atau None."
            )

        response = condition.get("response")

        if not isinstance(response, str):
            raise ValueError(
                "Field 'response' harus berupa string."
            )

        label = self.classifier.classify(response)

        if not isinstance(
            label,
            ClassificationLabel,
        ):
            raise TypeError(
                "Classifier harus mengembalikan "
                "ClassificationLabel."
            )

        classified = dict(condition)

        classified["classification"] = label.value

        return classified

    def classify_record(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise TypeError(
                "record harus berupa dictionary."
            )

        classified_record = dict(record)

        for condition_name in self.CONDITIONS:
            condition = record.get(condition_name)

            classified_record[condition_name] = (
                self._classify_condition(condition)
            )

        return classified_record

    def run(self) -> int:
        """
        Menjalankan classification untuk seluruh raw result.

        Returns:
            Jumlah record yang berhasil diproses.
        """
        count = 0

        for record in self.loader.iter_records():
            classified_record = self.classify_record(
                record
            )

            self.writer.write(
                classified_record
            )

            count += 1

        return count