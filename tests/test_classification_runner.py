import json
from pathlib import Path

from evaluation.classification import (
    ClassificationRunner,
    RuleBasedResponseClassifier,
)


def write_raw_record(
    path: Path,
    sample_id: str = "sample_001",
):
    record = {
        "sample_id": sample_id,
        "llm_text": {
            "model_name": "test-llm",
            "model_type": "llm",
            "input_modality": "text",
            "response": (
                "I'm sorry, but I can't help with that."
            ),
            "classification": None,
        },
        "vlm_text": {
            "model_name": "test-vlm",
            "model_type": "vlm",
            "input_modality": "text",
            "response": (
                "I cannot provide that information."
            ),
            "classification": None,
        },
        "vlm_image": {
            "model_name": "test-vlm",
            "model_type": "vlm",
            "input_modality": "image",
            "response": (
                "Here is some general information."
            ),
            "classification": None,
        },
    }

    path.write_text(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_classification_runner(
    tmp_path: Path,
):
    raw_path = tmp_path / "raw.jsonl"
    classified_path = tmp_path / "classified.jsonl"

    write_raw_record(raw_path)

    classifier = RuleBasedResponseClassifier()

    runner = ClassificationRunner(
        classifier=classifier,
        input_path=raw_path,
        output_path=classified_path,
    )

    count = runner.run()

    assert count == 1
    assert classified_path.exists()

    lines = [
        line
        for line in classified_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(lines) == 1

    result = json.loads(lines[0])

    assert result["sample_id"] == "sample_001"

    assert (
        result["llm_text"]["classification"]
        == "refusal"
    )

    assert (
        result["vlm_text"]["classification"]
        == "refusal"
    )

    assert (
        result["vlm_image"]["classification"]
        == "ambiguous"
    )


def test_classification_preserves_response(
    tmp_path: Path,
):
    raw_path = tmp_path / "raw.jsonl"
    classified_path = tmp_path / "classified.jsonl"

    write_raw_record(raw_path)

    original = json.loads(
        raw_path.read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )

    runner = ClassificationRunner(
        classifier=RuleBasedResponseClassifier(),
        input_path=raw_path,
        output_path=classified_path,
    )

    runner.run()

    classified = json.loads(
        classified_path.read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )

    assert (
        classified["llm_text"]["response"]
        == original["llm_text"]["response"]
    )

    assert (
        classified["vlm_text"]["response"]
        == original["vlm_text"]["response"]
    )

    assert (
        classified["vlm_image"]["response"]
        == original["vlm_image"]["response"]
    )


def test_missing_raw_file(
    tmp_path: Path,
):
    raw_path = tmp_path / "missing.jsonl"
    classified_path = tmp_path / "classified.jsonl"

    runner = ClassificationRunner(
        classifier=RuleBasedResponseClassifier(),
        input_path=raw_path,
        output_path=classified_path,
    )

    try:
        runner.run()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Seharusnya FileNotFoundError."
        )