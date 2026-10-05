import json
from pathlib import Path

from evaluation.raw_result_writer import RawExperimentResultWriter
from evaluation.result import ExperimentResult
from evaluation.three_condition_result import ThreeConditionResult


def make_result(
    sample_id: str,
    *,
    model_name: str,
    model_type: str,
    input_modality: str,
    response: str,
    image_path=None,
):
    return ExperimentResult(
        sample_id=sample_id,
        benchmark="JBB-Behaviors",
        behavior="test behavior",
        category="test category",
        model_name=model_name,
        model_type=model_type,
        input_modality=input_modality,
        prompt="test prompt",
        image_path=image_path,
        response=response,
        classification=None,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        device="cpu",
        torch_dtype=None,
        metadata={
            "source_prompt": "test prompt",
        },
    )


def test_writer_creates_jsonl(tmp_path: Path):
    output_path = tmp_path / "results.jsonl"

    writer = RawExperimentResultWriter(output_path)

    result = ThreeConditionResult(
        sample_id="sample_001",
        llm_text=make_result(
            "sample_001",
            model_name="test-llm",
            model_type="llm",
            input_modality="text",
            response="LLM response",
        ),
        vlm_text=make_result(
            "sample_001",
            model_name="test-vlm",
            model_type="vlm",
            input_modality="text",
            response="VLM text response",
        ),
        vlm_image=make_result(
            "sample_001",
            model_name="test-vlm",
            model_type="vlm",
            input_modality="image",
            response="VLM image response",
            image_path=Path("image.png"),
        ),
    )

    writer.write(result)

    assert output_path.exists()

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    data = json.loads(lines[0])

    assert data["sample_id"] == "sample_001"

    assert data["llm_text"] is not None
    assert data["vlm_text"] is not None
    assert data["vlm_image"] is not None

    assert data["llm_text"]["response"] == "LLM response"
    assert data["vlm_text"]["response"] == "VLM text response"
    assert data["vlm_image"]["response"] == "VLM image response"

    assert data["vlm_image"]["image_path"] == "image.png"


def test_writer_preserves_raw_response(tmp_path: Path):
    output_path = tmp_path / "results.jsonl"

    writer = RawExperimentResultWriter(output_path)

    result = ThreeConditionResult(
        sample_id="sample_002",
        llm_text=make_result(
            "sample_002",
            model_name="test-llm",
            model_type="llm",
            input_modality="text",
            response="RAW LLM RESPONSE",
        ),
    )

    writer.write(result)

    data = json.loads(
        output_path.read_text(encoding="utf-8").splitlines()[0]
    )

    assert data["llm_text"]["response"] == "RAW LLM RESPONSE"

    # Classification belum dilakukan pada tahap raw result.
    assert data["llm_text"]["classification"] is None


def test_writer_appends_records(tmp_path: Path):
    output_path = tmp_path / "results.jsonl"

    writer = RawExperimentResultWriter(output_path)

    result_1 = ThreeConditionResult(
        sample_id="sample_001",
        llm_text=make_result(
            "sample_001",
            model_name="test-llm",
            model_type="llm",
            input_modality="text",
            response="response 1",
        ),
    )

    result_2 = ThreeConditionResult(
        sample_id="sample_002",
        llm_text=make_result(
            "sample_002",
            model_name="test-llm",
            model_type="llm",
            input_modality="text",
            response="response 2",
        ),
    )

    writer.write(result_1)
    writer.write(result_2)

    lines = [
        line
        for line in output_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(lines) == 2

    assert json.loads(lines[0])["sample_id"] == "sample_001"
    assert json.loads(lines[1])["sample_id"] == "sample_002"