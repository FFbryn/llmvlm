import json
from pathlib import Path

from benchmark_data.schema import BenchmarkSample
from evaluation.batch_experiment_runner import BatchExperimentRunner
from evaluation.result import ExperimentResult
from evaluation.three_condition_result import ThreeConditionResult


class DummyRunner:
    """
    Runner dummy untuk menguji mekanisme batch
    tanpa memuat model sebenarnya.
    """

    def __init__(self):
        self.run_count = 0

    def run_sample_loaded(self, sample_id):
        self.run_count += 1

        return make_three_condition_result(sample_id)


class DummySession:
    def __init__(self, runner):
        self.runner = runner

    def __enter__(self):
        return self.runner

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False


class DummyExecutor:
    """
    Executor dummy yang mengikuti API aktual
    ExperimentExecutor.
    """

    def __init__(self):
        self.runner = DummyRunner()

    def batch_session(self):
        return DummySession(self.runner)

    def run_sample_loaded(
        self,
        runner,
        sample_id,
    ):
        return runner.run_sample_loaded(sample_id)


def make_experiment_result(
    sample_id: str,
    model_name: str,
    model_type: str,
    input_modality: str,
    response: str,
) -> ExperimentResult:
    """
    Membuat ExperimentResult valid untuk kebutuhan test.
    """

    return ExperimentResult(
        sample_id=sample_id,
        benchmark="JBB-Behaviors",
        behavior="test behavior",
        category="test category",
        model_name=model_name,
        model_type=model_type,
        input_modality=input_modality,
        prompt=f"test prompt {sample_id}",
        image_path=None,
        response=response,
        classification=None,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
        device="cpu",
        torch_dtype=None,
        metadata={
            "source_prompt": f"test prompt {sample_id}",
        },
    )


def make_three_condition_result(
    sample_id: str,
) -> ThreeConditionResult:
    """
    Membuat ThreeConditionResult lengkap.
    """

    return ThreeConditionResult(
        sample_id=sample_id,

        llm_text=make_experiment_result(
            sample_id=sample_id,
            model_name="test-llm",
            model_type="llm",
            input_modality="text",
            response="LLM response",
        ),

        vlm_text=make_experiment_result(
            sample_id=sample_id,
            model_name="test-vlm",
            model_type="vlm",
            input_modality="text",
            response="VLM text response",
        ),

        vlm_image=make_experiment_result(
            sample_id=sample_id,
            model_name="test-vlm",
            model_type="vlm",
            input_modality="image",
            response="VLM image response",
        ),
    )


def make_sample(sample_id: str) -> BenchmarkSample:
    """
    Membuat BenchmarkSample dummy.

    Tidak membutuhkan dataset JBB sebenarnya.
    """

    return BenchmarkSample(
        sample_id=sample_id,
        benchmark="JBB-Behaviors",
        prompt=f"test prompt {sample_id}",
        category="test category",
        metadata={},
    )


def test_resume_skips_completed_samples(
    tmp_path: Path,
):
    output_path = tmp_path / "results.jsonl"

    # ---------------------------------------------------------
    # Simulasikan sample_001 sudah pernah selesai.
    # ---------------------------------------------------------

    output_path.write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    executor = DummyExecutor()

    batch_runner = BatchExperimentRunner(
        executor=executor,
        output_path=output_path,
        resume=True,
    )

    samples = [
        make_sample("sample_001"),
        make_sample("sample_002"),
        make_sample("sample_003"),
    ]

    summary = batch_runner.run(samples)

    # ---------------------------------------------------------
    # sample_001 harus di-skip.
    #
    # sample_002 dan sample_003 harus dijalankan.
    # ---------------------------------------------------------

    assert executor.runner.run_count == 2

    assert summary["total"] == 3
    assert summary["skipped"] == 1
    assert summary["completed"] == 2
    assert summary["failed"] == 0

    # ---------------------------------------------------------
    # Pastikan JSONL berisi tiga record.
    # ---------------------------------------------------------

    lines = [
        line
        for line in output_path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]

    assert len(lines) == 3

    sample_ids = [
        json.loads(line)["sample_id"]
        for line in lines
    ]

    # Tidak boleh terjadi duplicate sample_001.
    assert sample_ids.count("sample_001") == 1

    assert sample_ids.count("sample_002") == 1
    assert sample_ids.count("sample_003") == 1


def test_resume_false_reprocesses_completed_samples(
    tmp_path: Path,
):
    output_path = tmp_path / "results.jsonl"

    # ---------------------------------------------------------
    # sample_001 sudah tersedia dari eksperimen sebelumnya.
    # ---------------------------------------------------------

    output_path.write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    executor = DummyExecutor()

    batch_runner = BatchExperimentRunner(
        executor=executor,
        output_path=output_path,
        resume=False,
    )

    samples = [
        make_sample("sample_001"),
    ]

    summary = batch_runner.run(samples)

    # ---------------------------------------------------------
    # resume=False berarti sample tetap dijalankan.
    # ---------------------------------------------------------

    assert executor.runner.run_count == 1

    assert summary["total"] == 1
    assert summary["skipped"] == 0
    assert summary["completed"] == 1
    assert summary["failed"] == 0

    # ---------------------------------------------------------
    # Karena sample dijalankan ulang, akan ada dua
    # record dengan sample_id yang sama.
    # ---------------------------------------------------------

    lines = [
        line
        for line in output_path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]

    assert len(lines) == 2

    sample_ids = [
        json.loads(line)["sample_id"]
        for line in lines
    ]

    assert sample_ids.count("sample_001") == 2