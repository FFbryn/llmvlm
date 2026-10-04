import json
from pathlib import Path

from evaluation.batch_experiment_runner import BatchExperimentRunner
from evaluation.three_condition_result import ThreeConditionResult


class DummyRunner:
    def __init__(self):
        self.run_count = 0

    def run_sample_loaded(self, sample_id):
        self.run_count += 1

        return ThreeConditionResult(
            sample_id=sample_id
        )


class DummySession:
    def __init__(self, runner):
        self.runner = runner

    def __enter__(self):
        return self.runner

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class DummyExecutor:
    def __init__(self):
        self.runner = DummyRunner()

    def batch_session(self):
        return DummySession(self.runner)

    def run_sample_loaded(self, runner, sample_id):
        return runner.run_sample_loaded(sample_id)


class DummyWriter:
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write(self, result):
        with self.output_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    {
                        "sample_id": result.sample_id
                    }
                )
                + "\n"
            )


def test_resume_skips_completed_samples(tmp_path: Path):
    output_path = tmp_path / "results.jsonl"

    # Simulasikan bahwa sample_001 sudah selesai
    output_path.write_text(
        json.dumps(
            {
                "sample_id": "sample_001"
            }
        )
        + "\n",
        encoding="utf-8",
    )

    executor = DummyExecutor()

    runner = BatchExperimentRunner(
        executor=executor,
        output_path=output_path,
        sample_ids=[
            "sample_001",
            "sample_002",
            "sample_003",
        ],
        resume=True,
        writer_factory=DummyWriter,
    )

    runner.run()

    # sample_001 dilewati.
    assert executor.runner.run_count == 2

    lines = [
        line
        for line in output_path.read_text(
            encoding="utf-8"
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


def test_resume_false_reprocesses_completed_samples(tmp_path: Path):
    output_path = tmp_path / "results.jsonl"

    output_path.write_text(
        json.dumps(
            {
                "sample_id": "sample_001"
            }
        )
        + "\n",
        encoding="utf-8",
    )

    executor = DummyExecutor()

    runner = BatchExperimentRunner(
        executor=executor,
        output_path=output_path,
        sample_ids=[
            "sample_001",
        ],
        resume=False,
        writer_factory=DummyWriter,
    )

    runner.run()

    assert executor.runner.run_count == 1

    lines = [
        line
        for line in output_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    assert len(lines) == 2