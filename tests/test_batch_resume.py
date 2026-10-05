import json
from pathlib import Path

from benchmark_data.schema import BenchmarkSample
from evaluation.batch_experiment_runner import BatchExperimentRunner
from evaluation.three_condition_result import ThreeConditionResult


class DummyRunner:
    """
    Runner dummy untuk menguji batch lifecycle tanpa
    memuat model nyata.
    """

    def __init__(self):
        self.run_count = 0

    def run_sample_loaded(self, sample_id):
        self.run_count += 1

        return ThreeConditionResult(
            sample_id=sample_id,
            llm_text=object(),
            vlm_text=object(),
            vlm_image=object(),
        )


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
    BatchExperimentRunner.
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


def make_sample(sample_id: str) -> BenchmarkSample:
    """
    Membuat BenchmarkSample menggunakan schema
    benchmark yang sebenarnya.
    """

    return BenchmarkSample(
        sample_id=sample_id,
        benchmark="jbb",
        prompt=f"test prompt {sample_id}",
        category="test",
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

    runner = BatchExperimentRunner(
        executor=executor,
        output_path=output_path,
        resume=True,
    )

    samples = [
        make_sample("sample_001"),
        make_sample("sample_002"),
        make_sample("sample_003"),
    ]

    summary = runner.run(samples)

    # ---------------------------------------------------------
    # sample_001 harus di-skip.
    # sample_002 dan sample_003 harus dijalankan.
    # ---------------------------------------------------------

    assert executor.runner.run_count == 2

    assert summary["total"] == 3
    assert summary["skipped"] == 1
    assert summary["completed"] == 2
    assert summary["failed"] == 0

    # ---------------------------------------------------------
    # Pastikan tidak ada duplicate sample_001.
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

    assert sample_ids.count("sample_001") == 1
    assert sample_ids.count("sample_002") == 1
    assert sample_ids.count("sample_003") == 1


def test_resume_false_reprocesses_completed_samples(
    tmp_path: Path,
):
    output_path = tmp_path / "results.jsonl"

    # ---------------------------------------------------------
    # sample_001 sudah ada.
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

    runner = BatchExperimentRunner(
        executor=executor,
        output_path=output_path,
        resume=False,
    )

    samples = [
        make_sample("sample_001"),
    ]

    summary = runner.run(samples)

    # ---------------------------------------------------------
    # Karena resume=False, sample tetap dijalankan.
    # ---------------------------------------------------------

    assert executor.runner.run_count == 1

    assert summary["total"] == 1
    assert summary["skipped"] == 0
    assert summary["completed"] == 1
    assert summary["failed"] == 0

    # ---------------------------------------------------------
    # Karena resume=False memang mengulang eksperimen,
    # JSONL akan memiliki dua record dengan sample_id
    # yang sama.
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